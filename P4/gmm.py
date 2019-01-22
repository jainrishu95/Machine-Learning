import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
            self.means, membership, update_counter = kmeans.fit(x)
            Yik = np.zeros((x.shape[0], self.n_cluster))
            Yik[np.arange(x.shape[0]), membership] = 1
            Nk = np.reshape(np.sum(Yik, axis=0), (self.n_cluster, 1))
            self.pi_k = Nk / np.sum(Nk)
            self.variances = np.zeros((self.n_cluster, x.shape[1], x.shape[1]))
            for i in range(self.n_cluster):
                x_minus = x - self.means[i]
                Yi = np.reshape(Yik[:,i], (x.shape[0], 1))
                temp = Yi * x_minus
                num = np.dot(temp.T, x_minus)
                self.variances[i] = num / Nk[i]
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            self.means = np.random.rand(self.n_cluster, x.shape[1])
            self.variances = np.zeros((self.n_cluster, x.shape[1], x.shape[1]))
            for i in range(self.variances.shape[0]):
                self.variances[i] = np.identity(x.shape[1])
            self.pi_k = np.ones((self.n_cluster, )) * (1 / self.n_cluster)
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')

        log_likelihood = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
        update_counter = 0

        for j in range(self.max_iter):
            # E Step:
            Yik_new = self.Yik_new

            # M Step:
            Nk_new = np.reshape(np.sum(Yik_new, axis=0), (self.n_cluster, 1))
            means_new = np.dot(Yik_new.T, x) / Nk_new
            variances_new = np.zeros((self.n_cluster, x.shape[1], x.shape[1]))
            for i in range(self.n_cluster):
                x_minus = x - means_new[i]
                Yi = np.reshape(Yik_new[:, i], (x.shape[0], 1))
                temp = Yi * x_minus
                num = np.dot(temp.T, x_minus)
                variances_new[i] = num / Nk_new[i]
            pi_k_new = Nk_new / np.sum(Nk_new)
            self.means = means_new
            self.variances = variances_new
            self.pi_k = pi_k_new

            log_likelihood_new = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
            if abs(log_likelihood_new - log_likelihood) <= self.e:
                break
            else:
                update_counter += 1
                log_likelihood = log_likelihood_new

        self.pi_k = np.reshape(self.pi_k, (self.pi_k.shape[0],))
        return update_counter
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        k, d = self.means.shape
        samples = np.zeros((N, d))
        for di in range(N):
            random_val = np.random.multinomial(1, self.pi_k)
            max_ = np.argmax(random_val)
            samples[di] = np.random.multivariate_normal(self.means[max_], self.variances[max_])

        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')

        N, D = x.shape
        self.Yik_new = np.zeros((N, self.n_cluster))
        for k in range(self.n_cluster):
            x_minus = x - self.means[k]
            while np.linalg.matrix_rank(self.variances[k]) != D:
                self.variances[k] += 0.001 * np.identity(D)
            inv = np.linalg.inv(self.variances[k])
            denum = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(self.variances[k]))
            num = self.pi_k[k] / denum
            matrix = np.exp(-0.5 * np.sum(np.multiply(np.dot(x_minus, inv), x_minus), axis=1))
            self.Yik_new[:, k] = num * matrix

        log_likelihood = float(np.sum(np.log(np.sum(self.Yik_new, axis=1))))

        sum_ = np.reshape(np.sum(self.Yik_new, axis=1), (N, 1))
        self.Yik_new /= sum_

        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood

    class Gaussian_pdf():
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf __init__')
            D = self.mean.shape[1]
            while np.linalg.matrix_rank(self.variance) != D:
                self.variance += 0.001 * np.identity(D)
            self.inv = np.linalg.inv(self.variance)
            self.c = ((2 * np.pi) ** D) * np.linalg.det(self.variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self, x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf getLikelihood')
            x_minus = x - self.mean
            matrix = np.multiply(np.dot(x_minus, self.inv), x_minus)
            p = np.exp(-0.5 * matrix) / np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
