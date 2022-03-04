from gatgnn_V.data                   import *
from gatgnn_V.utils                  import *
from gatgnn_V.model                  import *
from gatgnn_V.pytorch_early_stopping import *


#---- get symbols here 
symbol   = ['k','na','cs','mg','rb','zn']
s_id     = 4
symb     = symbol[s_id]
homepath = os.path.expanduser('~/Downloads/DILANGA-DATA/'+symb+'electrodes/')

#---- check the cif files 
src_CIFS = os.path.expanduser(homepath+'CIFS/')
cifs     = [x for x in os.listdir(src_CIFS) if x.endswith('.cif')]

#---- read-in the csv
electrodes_file = homepath+'high_low.csv'
df              = pd.read_csv(electrodes_file)
DF              = pd.DataFrame()
DF['mpid']      = np.unique(df.values.reshape(-1))


#---- keep mpids with present .cif structures
valid_rows      = np.zeros(DF.shape[0])

for i  in range(DF.shape[0]):
    if DF.mpid[i]+'.cif' in cifs:
        valid_rows[i] = 1

DF['found'] = valid_rows
DF          = DF[DF.found==1][['mpid']]

base_csv        = pd.read_csv(os.path.expanduser('~/Downloads/cgcnn_data/cgcnn_data/id_prop.csv'),names=['mpid','F_ener'])
print(base_csv.F_ener)


# In[3]:


# DATASET READING


# In[4]:


#---- get ion-type
all_ions    = ['Al', 'Ca', 'Cs', 'K', 'Li', 'Mg', 'Na', 'Rb', 'Y', 'Zn']
iontype     = [i for i in all_ions if i.upper()==symb.upper()][0]

DF['ion']   = iontype
DF['Eform'] = 0.01

#---- keep needed columns
DF          = DF[['mpid','Eform']]


# In[5]:


# IMPORT IMPORTANT MODULES


# In[6]:


from MODULE.data_Eform             import *
from MODULE.model_Huserver         import *
from MODULE.pytorch_early_stopping import *
from MODULE.utils                  import *


# In[7]:


# SETTING UP CODE TO RUN ON GPU


# In[8]:


gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


# In[9]:


# DATA PARAMETERS + MODEL HYPERPARAMETERS


# In[10]:


#---- for the data
random_num      = 456;random.seed(random_num);np.random.seed(random_num) 
DF              = DF.sample(frac=1,random_state=random_num)

graph_fmtCHOICE = ['CGCNN','MEGNET']
NORMALIZER      = DATA_normalizer(base_csv.F_ener.values)
norm_action     = 'norm'
batch_size      = 64
graph_fmt       = graph_fmtCHOICE[1]
crystal_property= 'formation-energy'

if graph_fmt   == 'CGCNN':
    RSM         = {'radius':8,'step':0.2,'max_num_nbr':12}
else:
    RSM         = {'radius':4,'step':0.5,'max_num_nbr':16}
    
test_param      = {'batch_size':256, 'shuffle': False}
data_src        = graph_fmt
root_dir        = src_CIFS

#---- for the model
num_epochs      = 500
learning_rate   = 1e-3
heads           = 4

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]


# In[11]:


# DATALOADER/ TARGET NORMALIZATION


# In[12]:


CRYSTAL_DATA       = CIF_Dataset(DF,**RSM,root_dir=root_dir)

idx_list           = list(range(len(DF)))
random.shuffle(idx_list)

test_idx           = idx_list
testing_set        = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm_action, df=DF,src=data_src)


# In[13]:


# NEURAL-NETWORK
# net  = GAT_NET(heads, neurons=64*2, nl=3,global_attention='composition', edge_format=data_src).to(device)

# NEURAL-NETWORK
net  = GAT_NET(4,None,neurons=64*2,nl=3,xtra_layers=True,global_attention='composition',
region_technique='pseudo-random',concat_comp=False,edge_format=data_src,interpretation=False).to(device)


# In[14]:


# LOSS & OPTMIZER & SCHEDULER
criterion   = nn.SmoothL1Loss().cuda()    ; funct = torch_MAE
optimizer   = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 5e-3)
scheduler   = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)


# In[15]:


# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct,device)


# In[16]:


# LOAD TRAINED MODEl


# In[17]:


# print(torch.load(f'NEW-FOLDER_Apr4th/formation_E.pt',map_location=device).keys())
_model1 = f'MODELS/formE_MEGNET.pt'
_model2 = f'NEW-FOLDER_Apr4th/formation_E.pt'
_model3 = f'MODELS/formE_MEGNET2.pt'
_model4 = f'MODELS/formE_CGCNN.pt'
_model5 = f'MODELS/crystal-checkpoint.pt'

net.load_state_dict(torch.load(_model5,map_location=device))


# In[18]:


# PREDICTION PHASE


# In[19]:


test_loader    = torch_DataLoader(dataset=testing_set, **test_param)

true_label, pred_label = torch.tensor([]).to(device),torch.tensor([]).to(device)
testset_idx    = torch.tensor([]).to(device)
num_elements   = torch.tensor([]).to(device)
net.eval()

for data in test_loader:
    data        = data.to(device)
    with torch.no_grad():
        predictions = net(data)
        predictions = NORMALIZER.denorm(predictions)
    
    data_y          = NORMALIZER.denorm(data.y.float())   
    true_label      = torch.cat([true_label,data_y],dim=0)
    print(f'(batch --- :{data.y.shape[0]:4})','---',metrics.eval_func(predictions,data_y).item())
    pred_label      = torch.cat([pred_label,predictions.float()],dim=0)
    testset_idx     = torch.cat([testset_idx,data.the_idx],dim=0)
    num_elements    = torch.cat([num_elements,data.num_atoms],dim=0)
  


# In[ ]:


# OUTPUT PREDICTION
with open(homepath+symb+'_Eform_prediction.csv','w') as outfile:
    outfile.write("mpid,real_Eform,pred_Eform\n")
    counter = 0
    for idx in testset_idx.cpu().long().tolist():
        mpid,eform = CRYSTAL_DATA.full_data.iloc[idx]
        pred       = pred_label[counter]
        newline    = f'{mpid},{eform:.3f},{pred:.3f}\n'
        outfile.write(newline)
        counter+=1
outdf = pd.read_csv(homepath+symb+'_Eform_prediction.csv').sort_values(by=['pred_Eform'])
print(outdf)


# In[ ]:




