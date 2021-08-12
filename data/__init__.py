import importlib
import torch.utils.data

from data.voxtest_dataset import VOXTestDataset



def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt, wav=None):
    """Create a dataloader.

    Args:
        opt (argparse.Namespace): Options packaged as a Namespace object.
        wav (torch.Tensor, Optional): Wav data with 16 kHz sample rate as 1d Float32 Tensor.
    """
    instance = VOXTestDataset()
    instance.initialize(opt, wav=wav)

    print("dataset [%s] of size %d was created" %
            (type(instance).__name__, len(instance)))

    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=opt.isTrain,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )

    return dataloader


