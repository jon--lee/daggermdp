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