from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE, NNCFG
from unet import uNet

def _train(model, dataloader, optimizer, criterion, epoch_iter=50):

   train_losses = []
   for epoch in range(epoch_iter):
      epoch_loss = 0
      for batch_X, batch_y in dataloader:
         optimizer.zero_grad()
         output = model(batch_X)
         output = output.squeeze()
         batch_y = batch_y.squeeze()
         # output = torch.mean(output, dim=1)
         # print(f'Output ; {output}')
         loss = criterion(output, batch_y)
         loss.backward()
         optimizer.step()
         epoch_loss+= loss.item()
      avg_epoch_loss = (epoch_loss/len(dataloader))
      train_losses.append(avg_epoch_loss)
      print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

   return model,train_losses
   


def train(cfg):
   
   nncfg = NNCFG()
   nncfg.argParser(cfg)

   # print(cfg.TRAIN_DATA)
   hdf5_file = h5py.File(cfg.TRAIN_DATA, 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
   
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   # print(p_data)

   positive_data = np.concatenate((p_data , s_data))

   X = np.concatenate([positive_data, noise_data], axis=0)
   Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise
   # print(Y)
   # print(f"X shape: {X.shape}, Y shape: {Y.shape}")

   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float))
   dataloader = DataLoader(dataset, batch_size=nncfg.batch_size, shuffle=True)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = None

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      model = DNN().to(device)
      model.apply(InitWeights)
      #criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      criterion = nn.BCEWithLogitsLoss()
      model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)
   
   elif cfg.MODEL_TYPE == MODEL_TYPE.UNET:
      model = uNet(samples=cfg.SAMPLE_WINDOW_SIZE, in_channels=3, out_channels = 1).to(device)
      # As they have used cross-entropy in the paper, I am using the same
      # criterion = nn.CrossEntropyLoss()
      # criterion = nn.BCELoss()
      criterion = nn.BCEWithLogitsLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      model, train_losses = _train(model, dataloader, optimizer, criterion, epoch_iter=nncfg.epoch_count)

   
   cfg.MODEL_FILE_NAME = cfg.MODEL_PATH + model.model_id

   # Save the model
   torch.save({
      'model_state_dict': model.state_dict(),
      'model_id'        : model.model_id,  # Save model ID
      'epoch_count'     : nncfg.epoch_count,
      'learning_rate'   : nncfg.learning_rate,
      'batch_size'      : nncfg.batch_size,
      'optimizer'       : optimizer.__class__.__name__.lower(),
      'training_loss'   : train_losses
   }, cfg.MODEL_FILE_NAME + ".pt")

   plot_loss(train_losses, cfg.MODEL_FILE_NAME)
   cfg.MODEL_FILE_NAME += ".pt"
