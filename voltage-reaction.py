from gatgnn_V.data                   import *
from gatgnn_V.utils                  import *
from gatgnn_V.model                  import *
from gatgnn_V.pytorch_early_stopping import *


# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN-Voltage')
parser.add_argument('--mode', default='evaluation',
                    choices=['training','evaluation'],
                    help='mode choices to run. evaluation for model evaluation and training for training new model.')
parser.add_argument('--data_src', default='CGCNN',choices=['CGCNN','MEGNET','NEW'],
                    help='selection of the materials dataset to use (default: CGCNN)')

# MOST CRUCIAL MODEL PARAMETERS
parser.add_argument('--batch',default=128, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons',default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')

args = parser.parse_args(sys.argv[1:])
src_CIFS = 'DATA/CIFS/'
df       = get_dataset(src_CIFS)


# EARLY-STOPPING INITIALIZATION
early_stopping = EarlyStopping(patience=stop_patience, increment=1e-6,verbose=True)

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct,device)

if args.mode == 'training':
    train_loader1   = torch_DataLoader(dataset=training_set1,   **train_param)
    train_loader2   = torch_DataLoader(dataset=training_set2,   **train_param)

    valid_loader1   = torch_DataLoader(dataset=validation_set1, **valid_param) 
    valid_loader2   = torch_DataLoader(dataset=validation_set2, **valid_param) 

    for epoch in range(num_epochs):
        # TRAINING-STAGE
        net.train()       
        start_time       = time.time()
        
        for i,High_Low_data in enumerate(zip(train_loader1,train_loader2)):
            data0, data1  = High_Low_data
            data0, data1  = data0.to(device), data1.to(device)
            
            predictions  = net(data0, data1)
            train_label  = metrics.set_label('training',data0)
            loss         = metrics('training',predictions,train_label,1)
            _            = metrics('training',predictions,train_label,2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.training_counter+=1
        metrics.reset_parameters('training',epoch)
        # VALIDATION-PHASE
        net.eval()

        for i,High_Low_data in enumerate(zip(valid_loader1,valid_loader2)):
            data0, data1  = High_Low_data
            data0, data1  = data0.to(device), data1.to(device)
            with torch.no_grad():
                predictions  = net(data0, data1)
            valid_label        = metrics.set_label('validation',data0)
            _                  = metrics('validation',predictions,valid_label,1)
            _                  = metrics('validation',predictions, valid_label,2)
            
            metrics.valid_counter+=1

        metrics.reset_parameters('validation',epoch)
        scheduler.step()
        end_time         = time.time()
        e_time           = end_time-start_time
        metrics.save_time(e_time)
        
        # EARLY-STOPPING
        early_stopping(metrics.valid_loss2[epoch], net)
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
    shutil.copy2('MODELS/crystal-checkpoint.pt', f'MODELS/{crystal_property}.pt')


else:
    pass 
# # NEURAL-NETWORK
# gatgnn1 = GATGNN(heads, neurons=64, nl=4,global_attention='composition', edge_format=data_src)
# gatgnn2 = GATGNN(heads, neurons=64, nl=4,global_attention='composition', edge_format=data_src)
# net     = PREDICTOR(128,gatgnn1,gatgnn2,neurons=128).to(device)
