import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval, Dataset_in_the_wild_eval
from models.Hiercon import Model, CompatibleCombinedLoss
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm

def evaluate_accuracy(dev_loader, model, device, criterion):
    """
    Evaluate model accuracy on validation set
    
    Args:
        dev_loader: Validation data loader
        model: HierCon model
        device: Device to run on
        criterion: Loss function (CompatibleCombinedLoss)
    
    Returns:
        val_loss: Average validation loss
        acc: Validation accuracy
    """
    val_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader, desc='Validation'):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            # Forward pass with features for contrastive learning
            batch_out, features = model(batch_x, return_features=True)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            # Calculate combined loss
            batch_loss, _, _ = criterion(batch_out, features, batch_y)
            val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    acc = 100 * (num_correct / num_total)
    return val_loss, acc


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()
    for batch_x, utt_id in tqdm(data_loader, desc='Evaluating'):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
        with open(save_path, 'a+') as fh:
            for f, cm in zip(utt_id, batch_score.tolist()):
                fh.write(f'{f} {cm}\n')
    print(f'Scores saved to {save_path}')

def train_epoch(train_loader, model, optimizer, device, criterion):
    """
    Train model for one epoch
    
    Args:
        train_loader: Training data loader
        model: HierCon model
        optimizer: Optimizer
        device: Device to run on
        criterion: Loss function (CompatibleCombinedLoss)
    
    Returns:
        running_loss: Average training loss
        avg_cce_loss: Average cross-entropy loss component
        avg_contrastive_loss: Average contrastive loss component
    """
    running_loss = 0
    running_cce = 0
    running_contrastive = 0
    num_total = 0.0
    
    model.train()
    
    for batch_x, batch_y in tqdm(train_loader, desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # Forward pass with features for contrastive learning
        batch_out, features = model(batch_x, return_features=True)
        
        # Calculate combined loss (CE + Contrastive)
        batch_loss, cce_loss, contrastive_loss = criterion(batch_out, features, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        running_cce += (cce_loss.item() * batch_size)
        running_contrastive += (contrastive_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    avg_cce_loss = running_cce / num_total
    avg_contrastive_loss = running_contrastive / num_total

    return running_loss, avg_cce_loss, avg_contrastive_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HierCon: Hierarchical Contrastive Attention for Audio Deepfake Detection (WWW 2026)'
    )
    
    # Dataset paths
    parser.add_argument('--database_path', type=str, default='/path/to/your/database/', 
                        help='Path to ASVspoof database directory')
    parser.add_argument('--protocols_path', type=str, default='/path/to/your/database/', 
                        help='Path to protocols directory')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='combined')
    
    # HierCon-specific parameters
    parser.add_argument('--group_size', type=int, default=3, 
                        help='Number of layers per group for hierarchical attention')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, 
                        help='Weight for contrastive loss component')
    parser.add_argument('--contrastive_margin', type=float, default=0.5, 
                        help='Margin for contrastive loss')
    parser.add_argument('--distance_type', type=str, default='euclidean', 
                        choices=['euclidean', 'cosine'], help='Distance metric for contrastive loss')
    
    # Model checkpoint
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint for loading')
    parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved model')
    
    # Evaluation arguments
    parser.add_argument('--track', type=str, default='DF', choices=['LA', 'In-the-Wild', 'DF'], 
                        help='Evaluation track')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    parser.add_argument('--eval', action='store_true', default=False, help='Run in evaluation mode')
    
    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True, 
                        help='Use CUDNN deterministic mode (default: True)')    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False, 
                        help='Use CUDNN benchmark mode (default: False)')

    # RawBoost data augmentation parameters
    parser.add_argument('--algo', type=int, default=3, 
                        help='RawBoost algorithm: 0=None, 1=LnL_convolutive, 2=ISD_additive, 3=SSI_additive, '
                             '4=All(1+2+3), 5=1+2, 6=1+3, 7=2+3, 8=Parallel(1,2)')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5, help='Number of notch filters')
    parser.add_argument('--minF', type=int, default=20, help='Min centre frequency [Hz]')
    parser.add_argument('--maxF', type=int, default=8000, help='Max centre frequency [Hz]')
    parser.add_argument('--minBW', type=int, default=100, help='Min filter width [Hz]')
    parser.add_argument('--maxBW', type=int, default=1000, help='Max filter width [Hz]')
    parser.add_argument('--minCoeff', type=int, default=10, help='Min filter coefficients')
    parser.add_argument('--maxCoeff', type=int, default=100, help='Max filter coefficients')
    parser.add_argument('--minG', type=int, default=0, help='Min gain factor')
    parser.add_argument('--maxG', type=int, default=0, help='Max gain factor')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, help='Min gain difference')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, help='Max gain difference')
    parser.add_argument('--N_f', type=int, default=5, help='Order of (non-)linearity')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, help='Max uniformly distributed samples [%]')
    parser.add_argument('--g_sd', type=int, default=2, help='Gain parameter')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, help='Min SNR for coloured noise')
    parser.add_argument('--SNRmax', type=int, default=40, help='Max SNR for coloured noise')

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    # Make experiment reproducible
    set_random_seed(args.seed, args)
    track = args.track

    # Define model saving path
    model_tag = f'model_{track}_{args.loss}_{args.num_epochs}_{args.batch_size}_{args.lr}'
    if args.comment:
        model_tag += f'_{args.comment}'
    model_save_path = os.path.join('models', model_tag)

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Initialize HierCon model
    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'Model: HierCon with {nb_params:,} parameters')
    print(f'Group size: {args.group_size}, Contrastive weight: {args.contrastive_weight}')

    model = nn.DataParallel(model).to(device)

    # Set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize combined loss function (CE + Contrastive)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = CompatibleCombinedLoss(
        weight=weight,
        margin=args.contrastive_margin,
        contrastive_weight=args.contrastive_weight,
        distance_type=args.distance_type
    )
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded: {args.model_path}')

    # Evaluation mode - In-the-Wild dataset
    if args.track == 'In-the-Wild':
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print(f'No. of eval trials: {len(file_eval)}')
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        exit(0)

    # Evaluation mode - DF or LA dataset
    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        print(f'No. of eval trials: {len(file_eval)}')
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        exit(0)

    # Define train dataloader
    d_label_trn, file_train = genSpoof_list(
        dir_meta=os.path.join(args.protocols_path, 'ASVspoof_DF_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),
        is_train=True, is_eval=False)
    print(f'No. of training trials: {len(file_train)}')
    
    train_set = Dataset_ASVspoof2019_train(
        args, list_IDs=file_train, labels=d_label_trn,
        base_dir=os.path.join(args.database_path, 'ASVspoof2019_LA_train/'), algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    del train_set, d_label_trn

    # Training loop
    num_epochs = args.num_epochs
    writer = SummaryWriter(f'logs/{model_tag}')
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\n========== Epoch {epoch+1}/{num_epochs} ==========')
        
        running_loss, cce_loss, contrastive_loss = train_epoch(
            train_loader, model, optimizer, device, criterion
        )
        
        # Early stopping based on training loss
        if running_loss < best_val_loss:
            best_val_loss = running_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print(f'✓ Best model saved at epoch {epoch+1}')
        else:
            patience_counter += 1

        # Log to tensorboard
        writer.add_scalar('train/total_loss', running_loss, epoch)
        writer.add_scalar('train/cce_loss', cce_loss, epoch)
        writer.add_scalar('train/contrastive_loss', contrastive_loss, epoch)
        
        print(f'Epoch {epoch+1}: Loss={running_loss:.4f} (CE={cce_loss:.4f}, Contrastive={contrastive_loss:.4f})')
        
        # Save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}.pth'))

        # Early stopping
        if patience_counter >= 5:
            print(f'Early stopping at epoch {epoch+1}. Best model from epoch {epoch+1-patience_counter}')
            break
