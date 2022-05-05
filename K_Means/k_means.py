import numpy as np

class KMeans:
    def __init__(self,data,num_clustres):
        '''
        初始化函数
        :param data: 传入的数据
        :param num_clustres: K值，簇的个数
        '''
        self.data = data
        self.num_clustres = num_clustres

    def train(self,max_iterations):
        '''
        训练
        :param self:
        :param max_iterations: 迭代次数
        :return:中心点位置、样本点属于哪个簇
        '''
        #1.先随机选取k个中心点
        centroids = KMeans.centroids_init(self.data, self.num_clustres)
        #2.开始训练
        num_examples=self.data.shape[0]
        closest_centroid_ids=np.empty((num_examples,1))
        for _ in range(max_iterations):
            #3.得到当前每一个样本点属于哪个簇
            closest_centroid_ids = KMeans.centroids_find_closet(self.data,centroids)
            #4.进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data,closest_centroid_ids,self.num_clustres)
        return centroids, closest_centroid_ids

    @staticmethod
    def centroids_init(data, num_clustres):
        '''
        随机选取K个中心点
        :param self:
        :param data: 传进来的数据
        :param num_clustres: K值
        :return: 选取的K个中心点
        '''
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clustres],:]
        return centroids

    @classmethod
    def centroids_find_closet(self, data, centroids):
        '''
        计算最小距离
        :param data: 传入的数据
        :param centroids: 随机选取的随机点
        :return: 各个样本点属于哪个簇
        '''
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples,1))
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids,1))    #记录当前样本到各个中心点的距离
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index,:]-centroids[centroid_index,:]
                distance[centroid_index]=np.sum(distance_diff**2)    #算距离
            closest_centroids_ids[example_index]=np.argmin(distance)   #记录当前样本属于哪个簇
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroid_ids, num_clustres):
        '''
        进行中心点更新
        :param data: 样本数据
        :param closest_centroid_ids: 样本点属于哪个簇
        :param num_clustres: K值
        :return: 更新后的中点位置
        '''
        num_features = data.shape[1]
        centroids = np.zeros((num_clustres,num_features))
        for centroid_id in range(num_clustres):
            closest_ids = closest_centroid_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(),:],axis=0)
        return centroids