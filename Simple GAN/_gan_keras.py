import tensorflow as tf; print(tf.__version__)
from tensorflow.contrib import layers
import pandas as pd
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
from scipy import stats
import umap # for dimensionality reduction
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import copy

tf.reset_default_graph()
session = tf.InteractiveSession()

class GAN_keras:
    """
        GAN class. This class contains the implementation of Cell State GAN network.
        Parameters
        ----------
        kwargs:
        
        data_max_value: double
            number of gene expression space dimensions.
        gene_num: integer
            number of latent space dimensions.
        df: pandas data frame
            input data to be fitted.
    """

    def __init__(self, data_max_value, gene_num, df,**kwargs):
        
        
        self.data_max_value = data_max_value
        self.batch_size = 32
        self.disc_size = 200
        self.gen_size = 600
        self.lambda_val = 10
        self.gene_num = gene_num
        self.df = df.values
        self.g_train = []
        self.d_train = []
        self.z = []
        self.x_true = []
        self.x_generated = []
        self.pertvecs = {}
        self.init_network()
        
    
    def generator(self,z):
        """
        Generator network. This function contains the implementation of the generator part of
        the GAN.
        Parameters
        ----------
        input args:
        
        z: array
            noise vector.
            
        output args:
        
        gen_output: array
            output of the generator network, it has the same dimensionality
            as the original data.
            
        
        """
        with tf.variable_scope('generator'):
            gen_dense1 = tf.layers.dense(inputs=z, 
                units=self.gen_size, 
                activation=None, 
                name="gen_dense1")

            gen_dense1 = tf.nn.leaky_relu(gen_dense1)

            gen_dense2 = tf.layers.dense(inputs=gen_dense1, 
                units=self.gen_size, 
                activation=None, 
                name="gen_dense2")

            gen_dense2 = tf.nn.leaky_relu(gen_dense2) # gen dense2


            gen_output = tf.layers.dense(inputs=gen_dense2, 
                units=self.gene_num, 
                activation=None, 
                name="gen_output")
            gen_output = tf.nn.leaky_relu(gen_output)

            return gen_output

    def discriminator(self,x, reuse):
        """
        Discriminator network. This function contains the implementation of the discriminator part of
        the GAN.
        Parameters
        ----------
        input args:
        
        x: array
            Input vector. Has the same dimensionality as the input data.
            
        reuse: boolean
            Determines if the weights should be reused.
            
        output args:
        
        gen_output: double
            Wasserstein distance.
            
        
        """
        
        with tf.variable_scope('discriminator', reuse=reuse):
            disc_dense1 = tf.layers.dense(inputs=x, 
                units=self.disc_size, 
                activation=None, 
                name="disc_dense1")
            disc_dense1 = tf.nn.leaky_relu(disc_dense1)


            disc_dense2 = tf.layers.dense(inputs=disc_dense1, 
                units=self.disc_size, 
                activation=None, 
                name="disc_dense2")
            disc_dense2 = tf.nn.leaky_relu(disc_dense2)


            disc_output = tf.layers.dense(inputs=disc_dense2, 
                units=1, 
                activation=None, 
                name="disc_output")

            return disc_output



    def noise_prior(self,batch_size, dim):
        
        """
        Discriminator network. This function contains the implementation of the discriminator part of
        the GAN.
        Parameters
        ----------
        input args:
        
        x: array
            Input vector. Has the same dimensionality as the input data.
            
        reuse: boolean
            Determines if the weights should be reused.
            
        output args:
        
        gen_output: double
            Wasserstein distance.
            
        
        """
        
        temp_norm = np.random.normal(0.0, self.data_max_value/10, size=(self.batch_size, dim))
        temp_poisson = np.random.poisson(1, size=(batch_size, dim))
        
        return np.abs(temp_norm + temp_poisson)

    def init_network(self):
        """
        This function initializes the network using the parameters calculated from
        the input data frame.
        
        """
        gene_num = self.gene_num
        
        with tf.name_scope('placeholders'):
            self.x_true = tf.placeholder(tf.float32, shape=(None, gene_num))
            self.z = tf.placeholder(tf.float32, shape=(None, 100))


        self.x_generated = self.generator(self.z)
        
        d_true = self.discriminator(self.x_true, reuse=False)
        d_generated = self.discriminator(self.x_generated, reuse=True)

        with tf.name_scope('regularizer'):
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.x_true + (1 - epsilon) * self.x_generated
            d_hat = self.discriminator(x_hat, reuse=True)

            gradients = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=1))
            d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

        with tf.name_scope('loss'):
            g_loss = tf.reduce_mean(d_generated)
            d_loss = (tf.reduce_mean(d_true) - tf.reduce_mean(d_generated) +
                      self.lambda_val * d_regularizer)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)

            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            self.g_train = optimizer.minimize(g_loss, var_list=g_vars)
            d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
            self.d_train = optimizer.minimize(d_loss, var_list=d_vars)

        tf.global_variables_initializer().run()
   
    def train_all(self,n_iter):
        
        """
        This trains the network with the pre-defined parameters for n_iter times.
        Parameters
        ----------
        input args:
        
        n_iter: int
            Determines how many train steps should the network do. 
            It is not equivalent to epoch numbers.
            
        
        """
        
        i = 0
        while i < n_iter:
            
            if i % 100 == 0:
                print(str((i/n_iter)*100) + '%')
            idx = np.random.randint(self.df.shape[0], size=self.batch_size)

            x_data = self.df[idx,:]

            noise = self.noise_prior(self.batch_size, 100)

            session.run(self.g_train, feed_dict={self.z : noise})
            for j in range(5):
                session.run(self.d_train, feed_dict={self.x_true : x_data, self.z : noise})
            i += 1
            
        print("Training completed!")
        
    def plot_UMAP(self, n_cells):
        """
        This function generates n_cells, reduce the dimensionality with UMAP
        and plot the results.
        Parameters
        ----------
        input args:
        
        n_cells: int
            How many cells should it generate.
           
        """
        gen_cells = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
        gen_cells = pd.DataFrame(gen_cells)
        gen_cells = gen_cells.iloc[gen_cells.index==1]
        
        for cell in range(n_cells-1):
            out_gen_temp = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
            out_gen_temp = pd.DataFrame(out_gen_temp)
            out_gen_temp = out_gen_temp.iloc[out_gen_temp.index==1]
            gen_cells = gen_cells.append(out_gen_temp, ignore_index=True)
        
        df = gen_cells
        df_orig = pd.DataFrame(self.df)
        df = df.assign(lab = 1)
        df_orig = df_orig.assign(lab = 0)
        df_new = df_orig.append(df, ignore_index=True)
        
        pca = PCA(n_components=100)
        data_pca = pca.fit_transform(df_new.loc[:, df_new.columns != 'lab'])
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(data_pca)
        colors = df_new.loc[:,'lab']
        palette = np.array(sns.color_palette("hls", 2))
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(embedding[:,0], embedding[:,1], lw=0, s=40,c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        
        ax.axis('tight')
        plt.show()
        return([embedding[:,0], embedding[:,1], colors])
        
    def draw_cells(self, n_cells):
        """
        This function draws n_cells sample from the learned distribution.
        Parameters
        ----------
        input args:
        
        n_cells: array
            How many cells should it generate.
        
        """
        gen_cells = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
        gen_cells = pd.DataFrame(gen_cells)
        gen_cells = gen_cells.iloc[gen_cells.index==1]
        
        for cell in range(n_cells-1):
            out_gen_temp = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
            out_gen_temp = pd.DataFrame(out_gen_temp)
            out_gen_temp = out_gen_temp.iloc[out_gen_temp.index==1]
            gen_cells = gen_cells.append(out_gen_temp, ignore_index=True)
        
        
        return(gen_cells)
    
    def recover_latentrep(self, vec_input, num_run):
        """
        Given a real data point this function recovers its latent space representation.
        Parameters
        ----------
        input args:
        
        vec_input: array
            Input vector of the real data, e.g.: gene expression data.
            
        num_run: int
            Determines the length of the search for the latent vector.
            
        output args:
        
        Latent space vector.
            
        
        """
       
        cortest = []
        
        noise_test = []
        for cell in range(num_run):
            
            noise_saved_1 = self.noise_prior(1, 100)

            out_gen_temp = session.run(self.x_generated, {self.z : noise_saved_1})[0,:]
            

            dist = mean_squared_error(vec_input, out_gen_temp)
            cortest.append(dist)
            noise_test.append(noise_saved_1[0,:])
            
        return(noise_test[np.argmin(cortest)])
    
    def draw_cells_with_seed(self, latent_vec):
        """
        Given a latent representation vector, draw a sample.
        Parameters
        ----------
        input args:
        
        latent_vec: array
            Input vector of the real data, e.g.: gene expression data.
            
            
        output args:
        
        Generated data point.
            
        
        """
        
        latent_vec = pd.DataFrame(latent_vec.reshape(-1, len(latent_vec))).append([latent_vec]*self.batch_size,ignore_index=True)
        
        gen_cells = session.run(self.x_generated, {self.z : latent_vec})
        gen_cells = pd.DataFrame(gen_cells)
        gen_cells = gen_cells.iloc[gen_cells.index==1]
        
        return(gen_cells)
    
    def interpolate_cells(self, startvec, endvec, interpol_points):
        
        """
        Given two real data points calculates interpol_points number
        of generated samples between them.
        Parameters
        ----------
        input args:
        
        startvec: array
            Input vector of the real data, e.g.: gene expression data.
        endvec: array
            Input vector of the real data, e.g.: gene expression data.
        interpol_points: int
            How many interpolated points are there.
            
            
        output args:
        
        df_sim: data frame
            Data frame containing the simulated samples.
            
        
        """
        alphas = np.linspace(0,1,interpol_points+1)
        
        startvec = pd.DataFrame(startvec.reshape(-1, len(startvec))).append([startvec]*self.batch_size,ignore_index=True)
        endvec = pd.DataFrame(endvec.reshape(-1, len(endvec))).append([endvec]*self.batch_size,ignore_index=True)
        
        gen_ind = session.run(self.x_generated, {self.z : startvec})
        gen_ind = pd.DataFrame(gen_ind)
        gen_ind = gen_ind.iloc[gen_ind.index==1]
        df_sim = pd.DataFrame(gen_ind.values)
        for index in alphas:    
            
            noise_current = index*endvec + (1-index)*startvec
            
            gen_ind = session.run(self.x_generated, {self.z : noise_current})
            gen_ind = pd.DataFrame(gen_ind)
            gen_ind = gen_ind.iloc[gen_ind.index==1]
            
            df_sim = df_sim.append(gen_ind)
            #true_ind = true_ind.append(pandas.DataFrame((index*endvec + (1-index)*startvec)).T)
        return(df_sim)
    
    def train_pertvec(self, labels, states ,n_iter, sample_count, num_count):
        """
        Given two real data groups calculates the latent space difference vector between them.
        Parameters
        ----------
        input args:
        
        labels: data frame
            Input labels vector of the real data.
        group1: string
            Label of the group 1.
        group2: string
            Label of the group 2.
        
        celltype: string
            Label of the celltype.
            
        n_iter: int
            How long the training should last.
       
        sample_count: int
            How many real samples should the algorithm consider.
        num_count: int
            How long should the latent space recovery last.
            
            
        output args:
        
        pert_vecs_mean: data frame
            The means of the latent space difference vectors.
            
        
        """
        
        pert_vecs_raw = {}
        
        df_alldata = copy.deepcopy(self.df)
        
        i = 0
        while i < n_iter:
            
            if i % 100 == 0:
                print(str((i/n_iter)*100) + '%')
            if i % 1000 == 0:
                gen_cells = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
                gen_cells = pd.DataFrame(gen_cells)
                gen_cells = gen_cells.iloc[gen_cells.index==1]

                for cell in range(2-1):
                    out_gen_temp = session.run(self.x_generated, {self.z : self.noise_prior(1, 100)})
                    out_gen_temp = pd.DataFrame(out_gen_temp)
                    out_gen_temp = out_gen_temp.iloc[out_gen_temp.index==1]
                    gen_cells = gen_cells.append(out_gen_temp, ignore_index=True)

                df_g = gen_cells
                df_orig = pd.DataFrame(copy.deepcopy(df_alldata))
                df_g = df_g.assign(lab = 1)
                df_orig = df_orig.assign(lab = 0)
                df_new = df_orig.append(df_g, ignore_index=True)

                pca = PCA(n_components=100)
                data_pca = pca.fit_transform(df_new.loc[:, df_new.columns != 'lab'])
                reducer = umap.UMAP()
                embedding = reducer.fit_transform(data_pca)
                colors = df_new.loc[:,'lab']
                palette = np.array(sns.color_palette("hls", 2))
                f = plt.figure(figsize=(8, 8))
                ax = plt.subplot(aspect='equal')
                sc = ax.scatter(embedding[:,0], embedding[:,1], lw=0, s=40,c=palette[colors.astype(np.int)])
                plt.xlim(-25, 25)
                plt.ylim(-25, 25)

                ax.axis('tight')
                plt.show()
                
                
                
                
                print(str((i/n_iter)*100) + '%')
                
            idx = np.random.randint(df_alldata.shape[0], size=self.batch_size)

            x_data = df_alldata[idx,:]

            noise = self.noise_prior(self.batch_size, 100)

            session.run(self.g_train, feed_dict={self.z : noise})
            for j in range(5):
                session.run(self.d_train, feed_dict={self.x_true : x_data, self.z : noise})
            i += 1
            
        print("Training completed!")
        
        
        
        '''
        i = 0
        for index in range(sample_count): # do it with random sampling
            
            if i % 1 == 0:
                print(str((i/sample_count)*100) + '%')
            i += 1
            startvec = df_train_1.iloc[index,:]
            endvec = df_train_2.iloc[index,:]
            
            start_noise = self.recover_latentrep(startvec, num_count)
            end_noise = self.recover_latentrep(endvec, num_count)
            
            pert_vecs_raw[index] =  end_noise - start_noise
        
        pert_vecs_mean = pd.DataFrame(pert_vecs_raw).mean(axis=1)
        return(pert_vecs_mean)
        '''