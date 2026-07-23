import torch

import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import csv
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

from models import VanillaLeNet, CNSteerableLeNet, DNSteerableLeNet, DNRestrictedLeNet
from utils import *
from gradcam import *
from RadioGalaxyData.firstgalaxydata import FIRSTGalaxyData

# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)                        # Python built-in
    np.random.seed(seed)                     # NumPy
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed(seed)             # PyTorch GPU
    torch.cuda.manual_seed_all(seed)         # Multi-GPU
    torch.backends.cudnn.deterministic = True  # cuDNN deterministic ops
    torch.backends.cudnn.benchmark = False     # disable auto-tuner
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)   # Python hash randomness

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(42)
g = torch.Generator()
g.manual_seed(42)
# -----------------------------------------------------------------------------
# extract information from config file:

vars = parse_args()
config_dict, config = parse_config(vars['config'])

batch_size     = config_dict['training']['batch_size']
frac_val       = config_dict['training']['frac_val']
epochs         = config_dict['training']['epochs']
imsize         = config_dict['training']['imsize']
nclass         = config_dict['training']['num_classes']
learning_rate  = torch.tensor(config_dict['training']['lr0'])
weight_decay   = torch.tensor(config_dict['training']['decay'])

early_stopping = config_dict['model']['early_stopping']
quiet          = config_dict['model']['quiet']
nrot           = config_dict['model']['nrot']

csvfile        = config_dict['output']['csvfile']
modfile        = config_dict['output']['modfile']

config         = vars['config'].split('/')[-1][:-4]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Data loading:

crop     = transforms.CenterCrop(imsize)
pad      = transforms.Pad((0, 0, 1, 1), fill=0)
totensor = transforms.ToTensor()
normalise= transforms.Normalize((config_dict['data']['datamean'],), (config_dict['data']['datastd'],))

train_transform = transforms.Compose([
    crop,
    pad,
    transforms.RandomRotation(360, interpolation=transforms.InterpolationMode.BILINEAR, expand=False),
    totensor,
    normalise,
])
test_transform = transforms.Compose([
    crop,
    pad,
    totensor,
    normalise,
])


train_dataset = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], selected_split="train",
                    input_data_list=["galaxy_data_h5.h5"],is_PIL=True, is_RGB=False, transform=train_transform)

val_dataset = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], selected_split="valid",
                    input_data_list=["galaxy_data_h5.h5"],is_PIL=True, is_RGB=False, transform=train_transform)

test_dataset = locals()[config_dict['data']['dataset']](config_dict['data']['datadir'], selected_split="test",
                    input_data_list=["galaxy_data_h5.h5"],is_PIL=True, is_RGB=False, transform=test_transform)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=seed_worker
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    worker_init_fn=seed_worker
)
# -----------------------------------------------------------------------------

model = locals()[config_dict['model']['base']](1, nclass, imsize+1, kernel_size=5, N=nrot).to(device)

if not quiet:
    summary(model, (1, imsize+1, imsize+1))

# -----------------------------------------------------------------------------

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay.item())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, factor=0.9)

# -----------------------------------------------------------------------------
# training loop:
print("Training Loop")
rows = ['epoch', 'train_loss', 'val_loss', 'val_accuracy']
                        
with open(csvfile, 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)
 
best_loss = float("inf")
best_acc = 0.0
best_epoch = 0
patience = 50
patience_counter = 0
min_delta = 1e-4
for epoch in range(epochs):  # loop over the dataset multiple times
    
    train_loss = train(model, train_loader, optimizer, device)
    val_loss, val_acc = test(model, val_loader, device)
        
    scheduler.step(val_loss)

    # check early stopping criteria:
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        best_acc = val_acc
        best_epoch = epoch

        patience_counter = 0
        torch.save(model.state_dict(), modfile)

    else:
        patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
    # create output row:
    _results = [epoch, train_loss, val_loss, val_acc]
    
    with open(csvfile, 'a', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(_results)
            
    if not quiet:
        print('Epoch: {}, Validation Loss: {:4f}, Validation Accuracy: {:4f}'.format(epoch, val_loss, val_acc))
        print('Current learning rate is: {}'.format(optimizer.param_groups[0]['lr']))
        
if early_stopping:
    print(
        f"Best validation accuracy: {best_acc:.4f} "
        f"(loss={best_loss:.4f}) @ epoch {best_epoch}"
    )
    model.load_state_dict(torch.load(modfile))
else:
    print(f"Final validation accuracy: {val_acc:.4f}")

# -----------------------------------------------------------------------------
# create outputs:

if not early_stopping:
    torch.save(model.state_dict(), modfile)

# -----------------------------------------------------------------------------
# confusion matrix:

# model.load_state_dict(torch.load(modfile))
test_loss, test_acc = test(model, test_loader, device)
print("Final Test Accuracy:", test_acc)

cm, report = custom_cm(model, test_loader, device)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_dataset.selected_classes,
)

disp.plot(cmap="Blues")
plt.savefig("FIRST_dnlenet_cm.png")
plt.show()
print(report)
print(cm)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# END
