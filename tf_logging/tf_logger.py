import os
import errno
from tensorboardX import SummaryWriter
import torch

class Logger:

    def __init__(self, model_name, data_name, log_path):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_path, comment=self.comment)


    def log_acc(self, mode, acc, epoch):

        # var_class = torch.autograd.variable.Variable
        if isinstance(acc, torch.autograd.Variable):
            acc = acc.data.cpu().numpy()

        self.writer.add_scalar(
            '{}/acc'.format(mode + '_' + self.comment), acc, epoch)

    def log_loss(self, mode, loss, epoch):

        # var_class = torch.autograd.variable.Variable
        if isinstance(loss, torch.autograd.Variable):
            acc = loss.data.cpu().numpy()

        self.writer.add_scalar(
            '{}_loss'.format(mode + '_' + self.comment), loss, epoch)


    def display_status(self, epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
