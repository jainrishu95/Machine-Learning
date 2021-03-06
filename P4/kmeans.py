import numpy as np

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's
                assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeans class (filename: kmeans.py)')

        #intialize
        cluster_mean_indexes = np.random.choice(np.arange(N), self.n_cluster)
        cluster_mean = x[cluster_mean_indexes, :]
        update_counter = 0
        cluster_membership_index = 0
        J = 9223372036854775808

        for i in range(self.max_iter):
            A = np.reshape(np.sum(x ** 2, axis=1), (x.shape[0], 1))
            B = np.reshape(np.sum(cluster_mean ** 2, axis=1), (cluster_mean.shape[0], 1))
            AB = np.dot(x, cluster_mean.T)
            Matrix = A + B.T - 2 * AB
            cluster_membership_index = np.array(np.argmin(Matrix, axis=1))
            membership_binary = np.zeros((N, cluster_mean.shape[0]))
            membership_binary[np.arange(N), cluster_membership_index] = 1
            count = np.sum(membership_binary, axis=0)
            dist = np.multiply(Matrix, membership_binary)
            J_new = np.sum(np.sum(dist, axis=1)) / np.sum(count)
            if abs(J - J_new) <= self.e:
                break
            else:
                update_counter += 1
                cluster_mean = np.dot(membership_binary.T, x) / count.reshape((count.shape[0], 1))
                J = J_new

        # DONOT CHANGE CODE BELOW THIS LINE
        return (cluster_mean, cluster_membership_index, update_counter)

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, update_counter = kmeans.fit(x)
        centroid_labels = np.zeros((centroids.shape[0],))

        for i in range(self.n_cluster):
            indexes = np.argwhere(membership == i)[:,0]
            labels = y[indexes]
            centroid_labels[i] = np.argmax(np.bincount(labels))

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        # DONOT CHANGE CODE BELOW THIS LINE
        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # raise Exception(
        #     'Implement predict function in KMeansClassifier class (filename: kmeans.py)')

        # labels = np.zeros((x.shape[0],))
        # for i in range(x.shape[0]):
        #     labels[i] = self.centroid_labels[np.argmin(np.linalg.norm(x[i] - self.centroids, 2))]
        # return labels

        A = np.reshape(np.sum(x ** 2, axis=1), (x.shape[0], 1))
        B = np.reshape(np.sum(self.centroids ** 2, axis=1), (self.centroids.shape[0], 1))
        AB = np.dot(x, self.centroids.T)
        Matrix = A + B.T - 2 * AB
        cluster_membership_index = np.array(np.argmin(Matrix, axis=1))
        labels = self.centroid_labels[cluster_membership_index]

        # DONOT CHANGE CODE BELOW THIS LINE
        return labels