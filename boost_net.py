import tensorflow as tf
import numpy as np
from net import Net
from Net.tensor import gridnet, inputdata

class boostNet():
	'''
	A dimensionality reduction layer followed by multiple weak learners
	'''

	def __init__(self, net, layer, learner, phi=.5, iter_stop = 10, weak_iterations = 200, ensemble_name = 'Ensemble', delta_stop = 1e-10, path=''):
        self.ensemble_name = ensemble_name
        

        self.net = net # already constructed, a net with a convolutional layer
        self.conv_path = path # the path to the saved net weights
        self.layer = layer # the layer of the net which c

        self.weights = np.array([]) # a placeholder for the weights on the weak learners
        self.ensemble = [] # a placeholder for the weak learners

        self.delta_stop = delta_stop # gives a stopping criteria for the adabboost
        self.iter_stop = iter_stop # gives the number ofiterations of adaboost before stopping
        self.learner_class = learner_class # the class which is a subclass of learner.learner
        self.weak_iterations = 200 # number of iterations sent to each weak learner before training is terminated
        self.phi = phi # a hyperparameter of adaboost

    def train_conv(self, train_path, test_path, iterations=200):
        '''
        trains the convolutional net based on the input data
        '''
        data = inputdata(train_path, test_path) #input data's constructor may not be appropriate, add subclass if necessary
        if len(self.conv_path) == 0:
			self.conv_path = self.net.optimize(iterations,data, path = self.conv_path, batch_size=200)
		else:
			self.conv_path = self.net.optimize(iterations,data, path = self.conv_path, batch_size=200)

    def fit_class(self, train_path, test_path):
    	'''
    	performs the adaboost code using classification
    	'''
    	data = inputdata(train_path, test_path)

    	# convert the data to embedded space
    	sess = self.net.load(self.conv_path)
    	ims, labels = data.all_train_batch()
    	with sess.as_default():
	    	data_embedding = sess.run(self.layer, feed_dict={net.x:ims}) # try doing this all at once, might be too big
	    sess.close()

	    # data = [(emb, label) for emb, label in zip(data_embedding, labels)]
    	weak_learners = []
    	learner_weights = []
    	data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
    	i = 0
    	while i < self.iter_stop:
    		learner = self.learner_class.__init__() # instantiate a new weak learner
    		learner.optimize(self.weak_iterations, data_embedding, labels, data_weights)
    		learner.start()
    		err = learner.accuracy(data_embedding, labels, data_weights) 
    		# results = learner.accuracy(data_embedding) # results is expected to an array of classes
    		# raw_errors = np.equal(results, labels)
    		# err = np.sum((1-raw_errors)  * data_weights)

    		# get the weight of the learner
    		learner.stop()
    		learner_weight = np.log(phi*(1-err)/(err*(1-phi)))
    		learner_weights.append(learner_weight)

    		data_weights = data_weights*np.exp(-learner_weight*raw_errors)
    		normalization = np.sum(data_weights)
    		data_weights = data_weights/normalization

    		if err >= phi - self.delta_stop or err == 0:
    			break
    		i += 1
    		weak_learners.append(learner)

    	self.ensemble = weak_learners
    	self.weights = learner_weights/np.sum(np.array(learner_weights))

    def fit_reg(self, train_path, test_path):
    	'''
    	performs the adaboost algorithm using regression
    	'''
    	data = inputdata(train_path, test_path) # load in some data

    	# convert the data to embedded space
    	sess = self.net.load(self.conv_path) 
    	ims, labels = data.all_train_batch()
    	with sess.as_default():
	    	data_embedding = sess.run(self.layer, feed_dict={net.x:ims}) # try doing this all at once, might be too big
	    sess.close()

	    # data = [(emb, label) for emb, label in zip(data_embedding, labels)]
    	weak_learners = []
    	learner_weights = []
    	data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
    	i = 0
    	while i < self.iter_stop:
    		learner = self.learner_class.__init__() # instantiate a new weak learner

    		# train the weak learner based on the embeddings and weights
    		learner.optimize(self.weak_iterations, data_embedding, labels, data_weights)

    		# predict the training set 
    		self.learner.start()
    		results = learner.predict(data_embedding) # results is expected to an array of classes
    		self.learner.stop()

    		# get the weight of the learner
    		raw_error = learner.error(results, labels)
    		learner_weight = np.log(phi*(1-err)/(err*(1-phi)))
    		learner_weights.append(learner_weight)

    		data_weights = data_weights*np.exp(-learner_weight*raw_errors)
    		normalization = np.sum(data_weights)
    		data_weights = data_weights/normalization

    		if err >= phi - self.delta_stop or err == 0:
    			break
    		i += 1
    		weak_learners.append(learner)

    	self.ensemble = weak_learners
    	self.weights = learner_weights/np.sum(np.array(learner_weights))

    def run_conv(self, inputs):
    	sess = self.net.load(self.conv_path)
    	with sess.as_default():
	    	data_embedding = sess.run(self.layer, feed_dict={net.x:inputs}) # try doing this all at once, might be too big
	    sess.close()
	    return data_embedding

    def load_conv(self, net_path):
    	self.conv_path = net_path

    def get_weights(self):
        data = inputdata.GridData_UB(self.data,self.T)
        return data.get_weights(self.net)
    
    def predict_group(self, inputs):
    	# inputs is always a list/nparray
		values = np.zeros(len(inputs))
		inputs = self.run_conv(inputs)
		for weak_learner, weight in zip(self.ensemble, self.weights):
			weak_learner.start()
			values += weak_learner.predict(inputs) * self.weights
			weak_learner.stop()
		return values

	def predict(self, sess, inputs):
    	# single input
		value = 0.0
		for weak_learner, weight in zip(self.ensemble, self.weights):
			value += weak_learner.predict(inputs) * self.weights
		return values

    def return_stats(self):
        return self.net.get_stats()