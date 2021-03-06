from gatgnn_V.data                   import *
from gatgnn_V.utils                  import *
from gatgnn_V.model                  import *
from gatgnn_V.pytorch_early_stopping import *


# DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN-Voltage')
parser.add_argument('--mode', default='evaluation',
                    choices=['training','evaluation','cross-validation','CV'],
                    help='mode choices to run. evaluation for model evaluation, training for training new model, and cross-validation or CV for both training & evaluations.')
parser.add_argument('--graph_size', default='small',choices=['small','large'],
                    help='graph encoding format by neighborhood size, either 12 or 16')
parser.add_argument('--train_size',default=0.8, type=float,
                    help='ratio size of the training-set (default:0.8)')
parser.add_argument('--batch',default=128, type=int,
                    help='batch size to use within experinment  (default:128)')
parser.add_argument('--fold',default=10, type=int,
                    help='number of folds to set when using cross-validation mode (default:20)')

# MODEL PARAMETERS
parser.add_argument('--layers',default=3, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--neurons',default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')
parser.add_argument('--heads',default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--custom', action='store_false', help ='use custom reaction-voltage trained model')

args     = parser.parse_args(sys.argv[1:])

# EARLY-STOPPING INITIALIZATION
early_stopping = EarlyStopping(patience=100, increment=1e-6,verbose=True)

# GENERAL SETTINGS & INITIALIZATIONS
random_num  = 456;random.seed(random_num);np.random.seed(random_num) 
gpu_id      = 0
device      = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
criterion   = nn.HuberLoss().to(device)    ; funct = torch_MAE
metrics     = METRICS('voltage',criterion,funct,device)
num_epochs  = 1000
lr          = 5e-3
train_param = {'batch_size':args.batch, 'shuffle': True}
valid_param = {'batch_size':args.batch, 'shuffle': False}

# DATA
src_CIFS   = 'DATA/CIFS/CIFS-R/'
df         = get_dataset(src_CIFS)#.sample(frac=0.2)
idx_list   = list(range(len(df)))
random.shuffle(idx_list)
NORMALIZER = DATA_normalizer(df.avg_voltage.values)
norm       = 'bel'

# MODEL
if args.graph_size   == 'small':
    RSM         = {'radius':8,'step':0.2,'max_num_nbr':12}
else:
    RSM         = {'radius':4,'step':0.5,'max_num_nbr':16}
CRYSTAL_DATA    = CIF_Dataset(df,root_dir= src_CIFS,**RSM)

# NEURAL-NETWORK
if args.mode in ['training','evaluation']:
    gatgnn1 = GATGNN_R(args.heads, neurons=args.neurons, nl=args.layers, neighborHood=args.graph_size)
    gatgnn2 = GATGNN_R(args.heads, neurons=args.neurons, nl=args.layers, neighborHood=args.graph_size)
    model   = REACTION_PREDICTOR(args.neurons*2,gatgnn1,gatgnn2,neurons=args.neurons).to(device)


if  args.mode  == 'training':
    milestones         = [150,250]


    train_idx,test_val = train_test_split(idx_list,train_size=args.train_size,random_state=random_num)
    _, val_idx         = train_test_split(test_val,test_size=0.5,random_state=random_num)

    training_set1    = CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm,df=df,src=args.graph_size)
    training_set2    = CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm,df=df,src=args.graph_size,id_01=1)

    validation_set1  = CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size)
    validation_set2  = CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size,id_01=1)

    train_loader1   = torch_DataLoader(dataset=training_set1,   **train_param)
    train_loader2   = torch_DataLoader(dataset=training_set2,   **train_param)

    optimizer       = optim.AdamW(model.parameters(), lr = lr, weight_decay = 5e-4)
    scheduler       = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)

    valid_loader1   = torch_DataLoader(dataset=validation_set1, **valid_param) 
    valid_loader2   = torch_DataLoader(dataset=validation_set2, **valid_param) 

    for epoch in range(num_epochs):
        # TRAINING-STAGE
        model.train()       
        start_time       = time.time()

        metrics, model   = train(model,train_loader1,train_loader2,metrics,optimizer, device)
        metrics.reset_parameters('training',epoch)

        # VALIDATION-PHASE
        model.eval()
        metrics, model,_  = evaluate(model,valid_loader1,valid_loader2,metrics, device)
        metrics.reset_parameters('validation',epoch)
        scheduler.step()

        end_time         = time.time()
        e_time           = end_time-start_time
        metrics.save_time(e_time)
        
        # EARLY-STOPPING
        early_stopping(metrics.valid_loss2[epoch], model)
        flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
        if early_stopping.FLAG == True:    estop_val = flag_value
        else:
            estop_val        = '@best: saving model...'; best_epoch = epoch+1
        output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

        if early_stopping.early_stop:
            print("> Early stopping")
            break
    # SAVING MODEL
    print(f"> DONE TRAINING !")
    shutil.copy2(f'MODEL/voltage-checkpoint.pt','MODEL/custom-checkpoint.pt')


elif args.mode == 'evaluation':
    model.eval()
    if args.custom:
        model.load_state_dict(torch.load(f'MODEL/custom-checkpoint.pt',map_location=device))
    else:
        model.load_state_dict(torch.load(f'MODEL/avg-voltage.pt',map_location=device))        
    valid_param        = {'batch_size':args.batch, 'shuffle': False}

    _        ,test_val = train_test_split(idx_list,train_size=args.train_size,random_state=random_num)
    test_idx, _        = train_test_split(test_val,test_size=0.5,random_state=random_num)

    testing_set1  = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size)
    testing_set2  = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size,id_01=1)

    test_loader1  = torch_DataLoader(dataset=testing_set1,   **valid_param)
    test_loader2  = torch_DataLoader(dataset=testing_set2,   **valid_param)

    # EVALUATION-PHASE
    _, _, out_labels  = evaluate(model,test_loader1,test_loader2,metrics, device, with_labels=True)


    # DENORMALIZING LABEL
    out_labels             = [torch.cat(x,-1) for x in out_labels]
    true_label, pred_label = out_labels
    # error2                 = torch_MAE(true_label, pred_label)

    true_label, pred_label = NORMALIZER.denorm(true_label), NORMALIZER.denorm(pred_label)       
    true_label, pred_label =  true_label.cpu().numpy(), pred_label.cpu().numpy()

    error2                 = MAE(true_label, pred_label)

    # SAVING PLOTS
    parity_plot(true_label, pred_label,error2)

    # WRITING RESULTS TO FILE
    save_results_to_file(test_idx, CRYSTAL_DATA, pred_label, true_label)

elif args.mode in ['cross-validation','CV']:

    milestones = [150,250]
    k_folds    = KFold(n_splits=args.fold, shuffle=True, random_state=random_num)
    iteration  = 1

    ALL_errors = []

    # ----------------
    # FOLD ITERATION 
    # ----------------
    for train_val,test_idx in k_folds.split(idx_list):

        gatgnn1 = GATGNN_R(args.heads, neurons=args.neurons, nl=args.layers, neighborHood=args.graph_size)
        gatgnn2 = GATGNN_R(args.heads, neurons=args.neurons, nl=args.layers, neighborHood=args.graph_size)
        model   = REACTION_PREDICTOR(args.neurons*2,gatgnn1,gatgnn2,neurons=args.neurons).to(device)

        metrics        = METRICS('voltage',criterion,funct,device)
        early_stopping = EarlyStopping(patience=100, increment=1e-6,verbose=True)

        print(tabulate([[f'Iteration # {iteration}']],tablefmt='fancy_grid'))
        
        train_idx, val_idx = train_test_split(train_val, train_size=0.9, random_state=random_num)

        training_set1      = CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm,df=df,src=args.graph_size)
        training_set2      = CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm,df=df,src=args.graph_size,id_01=1)

        validation_set1    = CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size)
        validation_set2    = CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size,id_01=1)

        train_loader1      = torch_DataLoader(dataset=training_set1,   **train_param)
        train_loader2      = torch_DataLoader(dataset=training_set2,   **train_param)

        optimizer          = optim.AdamW(model.parameters(), lr = lr, weight_decay = 5e-4)
        scheduler          = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)

        valid_loader1      = torch_DataLoader(dataset=validation_set1, **valid_param) 
        valid_loader2      = torch_DataLoader(dataset=validation_set2, **valid_param) 

        # --------------
        # TRAINING-STAGE
        # --------------
        for epoch in range(num_epochs):
            model.train()       
            start_time       = time.time()

            metrics, model   = train(model,train_loader1,train_loader2,metrics,optimizer, device)
            metrics.reset_parameters('training',epoch)

            # VALIDATION-PHASE
            model.eval()
            metrics, model,_  = evaluate(model,valid_loader1,valid_loader2,metrics, device)
            metrics.reset_parameters('validation',epoch)
            scheduler.step()

            end_time         = time.time()
            e_time           = end_time-start_time
            metrics.save_time(e_time)
            
            # EARLY-STOPPING
            early_stopping(metrics.valid_loss2[epoch], model)
            flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
            if early_stopping.FLAG == True:    estop_val = flag_value
            else:
                estop_val        = '@best: saving model...'; best_epoch = epoch+1
            output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

            if early_stopping.early_stop:
                print("> Early stopping")
                break

        # ----------------
        # EVALUATION-STAGE
        # ----------------

        model.eval()
        model.load_state_dict(torch.load(f'MODEL/voltage-checkpoint.pt',map_location=device))
        valid_param        = {'batch_size':args.batch, 'shuffle': False}

        testing_set1 = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size)
        testing_set2 = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm, df=df,src=args.graph_size,id_01=1)

        test_loader1  = torch_DataLoader(dataset=testing_set1,   **valid_param)
        test_loader2  = torch_DataLoader(dataset=testing_set2,   **valid_param)

        _, _, out_labels  = evaluate(model,test_loader1,test_loader2,metrics, device, with_labels=True)

        # DENORMALIZING LABEL
        out_labels             = [torch.cat(x,-1) for x in out_labels]
        true_label, pred_label = out_labels

        true_label, pred_label = NORMALIZER.denorm(true_label), NORMALIZER.denorm(pred_label)       
        true_label, pred_label =  true_label.cpu().numpy(), pred_label.cpu().numpy()

        error2                 = MAE(true_label, pred_label)

        save_results_to_file(test_idx, CRYSTAL_DATA, pred_label, true_label,idx_k_fold=f'{iteration}-')
        ALL_errors.append(error2)
        iteration += 1
    
    # FINAL ERROR
    mean_kfold_error = np.mean(ALL_errors)
    std_kfold_errorr = np.std(ALL_errors)

    print(f'avg. error: {mean_kfold_error} +/- {std_kfold_errorr}')
    




    # train_idx,test_val = train_test_split(idx_list,train_size=args.train_size,random_state=random_num)
    # _, val_idx         = train_test_split(test_val,test_size=0.5,random_state=random_num)



