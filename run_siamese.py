import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from model import CNN_STRM
from siamese_nn import SiameseNetwork
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf


from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import video_reader
import random 

import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level = logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
    
# logger for training accuracies
train_logger = setup_logger('Training_accuracy', './runs_strm/train_snn_new_output.log')

# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', './runs_strm/eval_snn_new_output.log')    

#############################################
#setting up seeds
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
########################################################

def main():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN_STRM(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        """
        This function parses the command-line arguments for the script. It uses the argparse library to define and parse the arguments.

        Returns:
            args: A namespace containing the arguments provided to the script.
        """

        # Initialize the argument parser
        parser = argparse.ArgumentParser()

        # Define the command-line arguments
        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="ssv2", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=100020, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[75000])
        parser.add_argument("--num_test_tasks", type=int, default=100, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=100, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=100, help="Num dataloader workers.")
        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet50", help="method")
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=100, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2,3])
        parser.add_argument("--scratch", choices=["bc", "bp", "new"], default="new", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=7, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--test_model_only", type=bool, default=False, help="Only testing the model from the given checkpoint")

        # Parse the command-line arguments
        args = parser.parse_args()

        # Set the scratch directory based on the argument value
        if args.scratch == "bc":
            args.scratch = "/mnt/storage/home2/tp8961/scratch"
        elif args.scratch == "bp":
            args.num_gpus = 4
            # this is low because of RAM constraints for the data loader
            args.num_workers = 3
            args.scratch = "/work/tp8961"
        elif args.scratch == "new":
            args.scratch = "./datasets_and_splits/"

        # Check if a checkpoint directory is specified
        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        # Set the image size and transformer linear input dimension based on the method
        if (args.method == "resnet50") or (args.method == "resnet34"):
            args.img_size = 224
        if args.method == "resnet50":
            args.trans_linear_in_dim = 2048
        else:
            args.trans_linear_in_dim = 512

        # Set the dataset paths based on the dataset argument
        if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "splits/ssv2_OTAM/")
            args.path = os.path.join(args.scratch, "datasets/ssv2_256x256q5.zip")
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(args.scratch, "splits/kinetics_CMN")
            args.path = os.path.join(args.scratch, "datasets/kinetics_256x256q5.zip")
        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(args.scratch, "splits/ucf_ARN")
            args.path = os.path.join(args.scratch, "datasets/ucf_256x256q5.zip")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(args.scratch, "splits/hmdb_ARN")
            args.path = os.path.join(args.scratch, "datasets/hmdb_256x256q5.zip")

        # Save the arguments to a pickle file
        with open("args.pkl", "wb") as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

        # Return the parsed arguments
        return args


    def run(self):
        """
        This function runs the training loop for the model. It iterates over the tasks in the video loader, 
        trains the model on each task, and performs optimization. It also logs the training statistics and 
        saves the model checkpoints.

        """

        # Configure the TensorFlow session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        # Start the TensorFlow session
        with tf.compat.v1.Session(config=config) as session:

            # Initialize lists to store training accuracies and losses
            train_accuracies = []
            losses = []

            # Get the total number of training iterations
            total_iterations = self.args.training_iterations

            # Initialize the iteration counter
            iteration = self.start_iteration

            # If only testing the model, load the checkpoint and test the model
            if self.args.test_model_only:
                print("Model being tested at path: " + self.args.test_model_path)
                self.load_checkpoint()
                #self.model = SiameseNetwork(self.model).to(self.device)
                accuracy_dict = self.test(session, 1)
                print(accuracy_dict)

            # Iterate over the tasks in the video loader
            for task_dict in self.video_loader:

                # Break the loop if the iteration counter reaches the total number of iterations
                if iteration >= total_iterations:
                    break

                # Increment the iteration counter
                iteration += 1

                # Enable gradient computation
                torch.set_grad_enabled(True)

                # Train the model on the task and get the task loss and accuracy
                task_loss, task_accuracy = self.train_task(task_dict)

                # Append the task accuracy and loss to the respective lists
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # Perform optimization if the iteration counter is a multiple of the tasks per batch or if it's the last iteration
                if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Step the learning rate scheduler
                self.scheduler.step()

                # Print the training statistics if the iteration counter is a multiple of the print frequency
                if (iteration + 1) % self.args.print_freq == 0:
                    print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                        torch.Tensor(train_accuracies).mean().item()))
                    train_logger.info("For Task: {0}, the training loss is {1} and Training Accuracy is {2}".format(iteration + 1, torch.Tensor(losses).mean().item(),
                        torch.Tensor(train_accuracies).mean().item()))

                    # Compute the average training accuracy and loss
                    avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                    avg_train_loss = torch.Tensor(losses).mean().item()

                    # Reset the lists for training accuracies and losses
                    train_accuracies = []
                    losses = []

                # Save a model checkpoint if the iteration counter is a multiple of the save frequency and it's not the last iteration
                if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                    self.save_checkpoint(iteration + 1)

                # Test the model and print the test accuracies if the iteration counter is in the list of test iterations and it's not the last iteration
                if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.test(session, iteration + 1)
                    print(accuracy_dict)
                    self.test_accuracies.print(self.logfile, accuracy_dict)

            # Save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        # Close the log file
        self.logfile.close()


    def train_task(self, task_dict):
        """
        This function trains the model on a single task. It calculates the loss and accuracy for the task and 
        performs backpropagation based on the loss.

        Parameters:
        task_dict (dict): A dictionary containing the task data.

        Returns:
        task_loss (torch.Tensor): The loss for the task.
        task_accuracy (float): The accuracy for the task.
        """

        # Prepare the task
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        # Move the images and labels to the device
        context_images = context_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_images = target_images.to(self.device)

        # Run the model on the task
        distance = model(support_images, context_labels, target_images)

        # Compute the loss
        task_loss = loss_function(distance, target_labels, self.device)

        # Compute the accuracy for the task
        accuracy = compute_accuracy(distance, target_labels)

        # Perform backpropagation based on the loss
        task_loss.backward(retain_graph=False)

        # Return the loss and accuracy for the task
        return task_loss, task_accuracy


    def test(self, session, num_episode):
        """
        This function tests the model over a number of episodes. It calculates the accuracy and loss for each task and 
        returns a dictionary containing the average accuracy, confidence interval, and average loss for the dataset.

        Parameters:
        session (Session): The current TensorFlow session.
        num_episode (int): The number of episodes to test the model on.

        Returns:
        accuracy_dict (dict): A dictionary containing the average accuracy, confidence interval, and average loss for 
                            the dataset.
        """

        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient calculations
        with torch.no_grad():

            # Set the dataset to testing mode
            self.video_loader.dataset.train = False

            # Initialize the accuracy dictionary, accuracies list, losses list, and iteration counter
            accuracy_dict = {}
            accuracies = []
            losses = []
            iteration = 0

            # Get the name of the dataset
            item = self.args.dataset

            # Loop over the tasks in the video loader
            for task_dict in self.video_loader:

                # Break the loop if the number of tasks exceeds the specified number of test tasks
                if iteration >= self.args.num_test_tasks:
                    break

                # Increment the iteration counter
                iteration += 1

                # Prepare the task
                context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

                # Run the model on the task
                model_dict = self.model(context_images, context_labels, target_images)
                siamese_network = SiameseNetwork(self.model).to(self.device)
                distance = siamese_network(context_images, context_labels, target_images, target_labels)
                
                # Compute the loss
                loss = loss_function(distance, target_labels, self.device)

                # Get the logits from the model dictionary and move them to the device
                prediction_logits = model_dict['logits'].to(self.device)

                # Get the logits after applying the query-distance-based similarity metric on patch-level enriched features
                predictionost_pat = model_dict['logits_post_pat'].to(self.device)
                
                # Move the target labels to the device
                target_labels = target_labels.to(self.device)

                # Add the logits before computing the accuracy
                prediction_logits = prediction_logits + 0.1*predictionost_pat
                
                accuracy = compute_accuracy(prediction_logits, target_labels)

                # Log the testing loss and accuracy for the task
                eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1, loss.item(),
                        accuracy.item()))

                # Append the loss and accuracy to their respective lists
                losses.append(loss.item())    
                accuracies.append(accuracy.item())

            # Compute the average accuracy, confidence interval, and average loss
            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            loss = np.array(losses).mean()

            # Add the average accuracy, confidence interval, and average loss to the accuracy dictionary
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}

            # Log the testing loss and accuracy for the episode
            eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(num_episode, loss, accuracy))

            # Set the dataset back to training mode
            self.video_loader.dataset.train = True

        # Set the model back to training mode
        self.model.train()

        # Return the accuracy dictionary
        return accuracy_dict

    def prepare_task(self,task_dict):
        """
        Prepare the task data for the Siamese Network.

        Parameters:
        task_dict (dict): A dictionary containing task data.
        device (torch.device): The device to move the data to.

        Returns:
        tuple: Prepared support images, target images, support labels, target labels, real target labels, and class list.
        """
        context_images = task_dict['support_set'][0].to(self.device)
        target_images = task_dict['target_set'][0].to(self.device)
        context_labels = task_dict['support_labels'][0].to(self.device)
        target_labels = task_dict['target_labels'][0].type(torch.LongTensor).to(self.device)
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        gpu_device = 'cuda'
        if self.args.test_model_only:
            checkpoint = torch.load(self.args.test_model_path, torch.device('gpu' if torch.cuda.is_available() else 'cpu'))
        else:
           checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        state_dict = {k.replace('module.', ''):v for k,v in checkpoint['model_state_dict'].items()}
        if self.device == gpu_device:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

def loss_function(distance, target_labels, device):
    """
    Compute the contrastive loss based on the distance between support and target images.

    Parameters:
    distance (torch.Tensor): The computed distance between support and target images.
    target_labels (torch.Tensor): The labels of the target images.
    device (torch.device): The device to perform the computation on.

    Returns:
    torch.Tensor: The computed loss.
    """
    # Ensure target_labels has the same shape as distance
    if target_labels.dim() == 1:
        target_labels = target_labels.unsqueeze(0)

    loss = F.mse_loss(distance, target_labels.float().to(device))
    return loss

def compute_accuracy(prediction_logits, test_labels):
    """
    Compute the accuracy based on the prediction_logits and target labels.

    Parameters:
    prediction_logits (torch.Tensor): The computed prediction_logits between support and target images.
    target_labels (torch.Tensor): The labels of the target images.

    Returns:
    torch.Tensor: The computed accuracy.
    """
    averaged_predictions = torch.logsumexp(prediction_logits, dim=0)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

if __name__ == "__main__":
    main()
