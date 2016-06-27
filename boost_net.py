import tensorflow as tf
import numpy as np
from net import Net
from Net.tensor import gridnet, inputdata

class boostNet():
	'''
	A single convolutional layer followed by multiple 
	'''

	def __init__(self, net, layer, learner, phi=.5, iter_stop = 10, weak_iterations = 200, ensemble_name = 'Ensemble', delta_stop = 1e-10, path=''):
        self.ensemble_name = ensemble_name
        
        self.net = net #already constructed
        self.conv_path = path
        self.ensemble = np.array([])
        self.weights = np.array([])
        self.layer = layer

        self.delta_stop = delta_stop
        self.iter_stop = iter_stop
        self.learner_class = learner_class # would need a general learner class?
        self.weak_iterations = 200
        self.phi = phi

    def train_conv(self, train_path, test_path, iterations=200):
        data = inputdata(train_path, test_path) #input data's constructor may not be appropriate, add subclass if necessary
        if len(self.conv_path) == 0:
			self.conv_path = self.net.optimize(iterations,data, path = self.conv_path, batch_size=200)
		else:
			self.conv_path = self.net.optimize(iterations,data, path = self.conv_path, batch_size=200)

    def fit_class(self, train_path, test_path):
    	data = inputdata(train_path, test_path)

    	sess = self.net.load(self.conv_path)
    	ims, labels = data.all_train_batch()
    	with sess.as_default():
	    	data_embedding = sess.run(self.layer, feed_dict={net.x:ims}) # try doing this all at once, might be too big
	    sess.close()

	    data = [(emb, label) for emb, label in zip(data_embedding, labels)]
    	weak_learners = []
    	learner_weights = []
    	data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
    	i = 0
    	while i < self.iter_stop:
    		learner = self.learner_class.__init__() # instantiate a new weak learner
    		learner.optimize(self.weak_iterations, (data_embedding, labels), data_weights)
    		learner.start()
    		results = learner.accuracy(data_embedding) # results is expected to an array of classes
    		learner.stop()
    		raw_errors = np.equal(results, labels)
    		err = np.sum((1-raw_errors)  * data_weights)
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
    	data = inputdata(train_path, test_path)

    	sess = self.net.load(self.conv_path)
    	ims, labels = data.all_train_batch()
    	with sess.as_default():
	    	data_embedding = sess.run(self.layer, feed_dict={net.x:ims}) # try doing this all at once, might be too big
	    sess.close()

	    data = [(emb, label) for emb, label in zip(data_embedding, labels)]
    	weak_learners = []
    	learner_weights = []
    	data_weights = np.array([1.0/len(labels) for _ in range(len(labels))])
    	i = 0
    	self.learner.start()
    	while i < self.iter_stop:
    		learner = self.learner.learner() # instantiate a new weak learner
    		learner.optimize(self.weak_iterations, data, data_weights)
    		results = learner.predict(data_embedding) # results is expected to an array of classes
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

    	self.learner.stop()
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