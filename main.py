import os
import pickle
import matplotlib.pyplot as plt

from WGANGP import WGANGP
from Loader import load_RGBA_SOUL


# run params
# Data folder to generate (Warning!: Should be located in run folder)
DATA_NAME = 'RGBA_Survival_SONATA_Malfunction'
RUN_FOLDER = 'C:/~/Result/'
mode ='build'   # Option: 'build or load'

'''
with open(RUN_FOLDER+'/obj.pkl', 'rb') as file:
    history = pickle.load(file)

print(history)
'''

BATCH_SIZE = 64
IMAGE_HEIGHT = 100 #100
IMAGE_WIDTH =80 #80
x_train = load_RGBA_SOUL(DATA_NAME, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)
#plt.imshow((x_train[0][0][0]+1)/2)

gan = WGANGP(input_dim = (IMAGE_HEIGHT,IMAGE_WIDTH,4)
        , critic_conv_filters = [64,128,256,512] # 차례대로 Conv filters 값 > 총 4개 필터 사용
        , critic_conv_kernel_size = [5,5,5,5]
        , critic_conv_strides = [2,2,2,2]
        , critic_batch_norm_momentum = None
        , critic_activation = 'leaky_relu'
        , critic_dropout_rate = None
        , critic_learning_rate = 0.0002
        , generator_initial_dense_layer_size = (5, 4, 512)
        , generator_upsample = [1,1,1,1] #1이면 conv 2 transpose, 2면 upsampling
        , generator_conv_filters = [256,128,64,4] # 차례로 Generator filters 값 > 총 4개 필터 사용
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [2,5,2,1]
        , generator_batch_norm_momentum = None
        , generator_activation = 'leaky_relu'
        , generator_dropout_rate = None
        , generator_learning_rate = 0.0002
        , optimiser = 'adam'
        , grad_weight = 10
        , z_dim = 100
        , batch_size = BATCH_SIZE
        )

if mode == 'build':
    gan.save(RUN_FOLDER)
else:
    gan.load_weights(RUN_FOLDER+'weights/weights.h5')

# model Critics
# model1 Generator
# model2 Critic_model
# model3 Critic+Generator model

#Training
EPOCHS = 10000
PRINT_EVERY_N_BATCHES = 10
N_CRITIC = 5
BATCH_SIZE = 64

gan.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , n_critic = N_CRITIC
    , using_generator = True
)
fig = plt.figure()
plt.plot([x[0] for x in gan.d_losses], 'Black',linestyle=':', label='Total loss') #  total loss
plt.plot([x[1] for x in gan.d_losses], 'g-.', label='Real data loss') #  Real data loss (w)
plt.plot([x[2] for x in gan.d_losses], 'r-', label='Fake data loss')   #  Fake data loss (w)
plt.plot([x[3] for x in gan.d_losses], 'b--', label='Gradient penalty loss')  # Gradient penalty loss (gp)

plt.plot(gan.g_losses, color='orange', label='Generator loss')                # Generator w loss (w)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title("Loss graph")
plt.legend()
plt.savefig('LOSS_graph.png', dpi=300)
plt.close()

#plt.xlim(0, 2000)
#plt.ylim(-1, 2)
