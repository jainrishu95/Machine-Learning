'''
        #loop to calculate means
        for i in range(self.max_iter):
            #calculate relation (r matrix)
            r = []
            for j in range(N):
                min_dist = 9223372036854775808
                min_cluster_index = -1
                for k in range(cluster_mean.shape[0]):
                    dist = np.linalg.norm(x[j] - cluster_mean[k])
                    if dist < min_dist:
                        min_cluster_index = k
                        min_dist = dist
                rik = list(map(lambda x: [0, 1][x == min_cluster_index], range(cluster_mean.shape[0])))
                r.append(rik)
            r = np.array(r)

            #calculating distortion measure
            J_new = 0
            for j in range(N):
                index = np.where(r[j] == 1)
                dist = np.linalg.norm(x[j] - cluster_mean[index])
                J_new += dist
            J_new /= N

            if abs(J - J_new) <= self.e:
                membership = []
                for j in range(len(r)):
                    index = np.where(r[j] == 1)
                    k = index[0][0] + 1
                    membership.append(k)
                return (cluster_mean, np.array(membership), update_counter)
            else:
                update_counter += 1
                J = J_new
                #calculate new means
                no_of_pts_in_clusters = np.sum(r, axis=0)
                for k in range(len(no_of_pts_in_clusters)):
                    if no_of_pts_in_clusters[k] == 0:
                        continue
                    sum_ = np.zeros((1, D))
                    for i in range(N):
                        if r[i][k] == 1:
                            sum_ += x[i]
                    cluster_mean[k] = sum_ / no_of_pts_in_clusters[k]

        membership = []
        for j in range(len(r)):
            index = np.where(r[j] == 1)
            k = index[0][0] + 1
            membership.append(k)
        '''