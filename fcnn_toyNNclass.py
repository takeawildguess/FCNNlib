class toyNN():
    def __init__(self, dataset='and', nb_pnt=20):
        """
        dataset: and, or, xor, stripe, triangle, square, circle
        nb_pnt: number of points to create for the dataset
        scale: horizontal domain -> x ranges from -scale to scale
        """
        self.dataset = dataset
        self.nb_pnt = nb_pnt
        self.scale = 2
        self.getPoints()
        self.getGrid()

    def groundTruth(self, XX):
        x1, x2 = XX[:, 0], XX[:, 1]
        if self.dataset == 'and':
            YY = ((x1>0) & (x2>0)).astype(int).reshape(-1, 1)
        elif self.dataset == 'or':
            YY = ((x1>0) | (x2>0)).astype(int).reshape(-1, 1)
        elif self.dataset == 'xor':
            YY = ((x1*x2>0)).astype(int).reshape(-1, 1)
        elif self.dataset == 'stripe':
            YY = (np.abs(x1-x2)<=1).astype(int).reshape(-1, 1)
        elif self.dataset == 'square':
            YY = ((np.abs(x1)+np.abs(x2))<=1).astype(int).reshape(-1, 1)
        elif self.dataset == 'circle':
            YY = ((x1**2+x2**2)<=1).astype(int).reshape(-1, 1)
        elif self.dataset == 'prod':
            YY = ((x1*x2)/4).reshape(-1, 1)
        elif self.dataset == 'sumSquares':
            YY = ((x1**2+x2**2)/4).reshape(-1, 1)
        elif self.dataset == 'polynom':
            YY = ((x1**2-3*x1*x2-x2**2)/4).reshape(-1, 1)
        elif self.dataset == 'squares':
            YY = ((np.abs(x1)+np.abs(x2))).astype(int).reshape(-1, 1)
        elif self.dataset == 'circles':
            YY = ((x1**2+x2**2)/2).astype(int).reshape(-1, 1)
        elif self.dataset == 'quadrants':
            YY = (2*(np.arctan2(x2, x1)/np.pi+1)).astype(int).reshape(-1, 1)
        else:
            pass

        if self.dataset in ['prod', 'sumSquares', 'polynom']:
            self.kind = 'regr'
        elif self.dataset in ['squares', 'circles', 'quadrants']:
            self.kind = 'multiCls'
        elif self.dataset in ['and', 'or', 'xor', 'stripe', 'square', 'circle']:
            self.kind = 'binCls'
        else:
            pass

        return YY

    def getPoints(self,):
        XX = self.scale*(2*np.random.rand(self.nb_pnt, 2)-1)
        YY = self.groundTruth(XX)
        self.nb_class = np.unique(YY).shape[0] if self.kind in ['binCls', 'multiCls'] else None
        self.XX, self.YY = XX, YY

    def getGrid(self, np_gp=100):
        np_gp, mrg = np_gp+1, 1
        x1, x2 = self.XX[:, 0], self.XX[:, 1]
        x1_min, x1_max = x1.min() - mrg, x1.max() + mrg
        x2_min, x2_max = x2.min() - mrg, x2.max() + mrg
        self.Xgrd1, self.Xgrd2 = np.meshgrid(np.linspace(x1_min, x1_max, np_gp), np.linspace(x2_min, x2_max, np_gp))
        XXgrd = np.c_[self.Xgrd1.ravel(), self.Xgrd2.ravel()]
        self.YYgrd = self.groundTruth(XXgrd)
        if self.kind == 'multiCls':
            self.YYgrd = np.minimum(self.nb_class-1, self.YYgrd)
        self.XXgrd = XXgrd
        self.x1, self.x2 = x1, x2

    def plotPoints(self, idx=None, figsize=(6, 6)):
        x1, x2 = self.XX[:, 0], self.XX[:, 1]
        plt.figure(figsize=figsize)
        plt.scatter(x1, x2, c=self.YY.reshape(-1), cmap=plt.cm.Spectral, alpha=.6)
        if idx is not None:
            x01, x02 = self.XX[idx, 0], self.XX[idx, 1]
            plt.gca().add_patch(plt.Circle((x01, x02), radius=.1, color='k', fill=False))
            plt.title('Sample ({:.2f}, {:.2f}) from class {:.0f}'.format(x01, x02, self.YY[idx,0]))
        plt.ylabel('$x_2$')
        plt.xlabel('$x_1$')

    def plotModelEstimate(self, np_gp=100, figsize=(6, 6)):
        # Plot the contour and training examples
        plt.figure(figsize=figsize)
        plt.subplot(211)
        plt.contourf(self.Xgrd1, self.Xgrd2, self.nn_Ygrd.reshape(self.Xgrd1.shape), cmap=plt.cm.Spectral, alpha=.4)
        plt.scatter(self.x1, self.x2, c=self.YY.reshape(-1), cmap=plt.cm.Spectral, alpha=.8)
        plt.ylabel('$x_2$')
        plt.xlabel('$x_1$')
        plt.title(self.mdlDescription())

    def toOneHotEncoding(self, YY):
        # transform the response variable to an one-hot-encoding representation
        if self.kind == 'multiCls':
            YYohe = np_utils.to_categorical(YY, num_classes=self.nb_class)
            return YYohe
        else:
            return YY

    def train(self, lib='tf', nb_epochs=100, dims=[2], activation='sigmoid', lr=.005, batchSize=100, opt='adam', display=False):
        """
        lib: tf (tensorflow), ks (keras), skl (scikit-learn), pt (pytorch)
        """
        self.lib = lib
        self.LR = lr
        self.nb_epochs = nb_epochs
        self.batchSize = batchSize
        self.activation = activation
        self.display = display
        self.opt = opt
        self.dims = [2] + dims + [self.nb_class if self.kind=='multiCls' else 1]
        self.nb_layer = len(self.dims)-1 # number of network layers with learnable parameters
        self.lastActFun = 'sigmoid' if self.kind == 'binCls' else 'softmax' \
            if self.kind == 'multiCls' else 'linear'
        if self.lib=='skl':
            self.sklModel()
        elif self.lib=='tf':
            self.tfModel()
            self.tfTraining()
        elif self.lib=='ks':
            self.kerasModel()
        elif self.lib=='pt':
            self.pytorchModel()
        else:
            print("Please select one of these libraries: tf (tensorflow), ks (kera), skl (scikit-learn)!")

        unitLabel = '-'.join([str(u) for u in self.dims])
        self.descrs = {'lib': self.lib, 'units': unitLabel, 'depth': str(len(dims)),\
            'act': activation, 'lr': lr, 'opt': opt,\
            'epochs': nb_epochs, 'batchSize': batchSize}

    def mdlDescription(self, keys=None):
        if not keys:
            keys = self.descrs.keys()
        return ', '.join(['{}: {}'.format(kk, vv) for kk, vv in self.descrs.items() if kk in keys])

    def sklModel(self):
        if self.opt in ['sgd', 'adam']:
            optName = self.opt
        else:
            print("this optimizer is not available in SciKitLearn!")
            optName = 'sgd'
        if self.kind == 'regr':
            mdl = MLPRegressor(hidden_layer_sizes=tuple(self.dims[1:-1]), max_iter=self.nb_epochs,\
                                alpha=0, activation=self.activation, learning_rate_init=self.LR,\
                                solver=optName, tol=1e-24)
        elif self.kind in ['binCls', 'multiCls']:
            mdl = MLPClassifier(hidden_layer_sizes=tuple(self.dims[1:-1]), max_iter=self.nb_epochs,\
                                alpha=0, activation=self.activation, learning_rate_init=self.LR,\
                                solver=optName, tol=1e-24)
        else:
            mdl = None
        if self.kind in ['regr', 'binCls']:
            YY = self.YY.ravel()
        else:
            YY = self.YY
        YY = self.YY.ravel()
        mdl.fit(self.XX, YY)
        self.nn_Ygrd = mdl.predict(self.XXgrd)
        self.lossHistory = mdl.loss_curve_
        self.nn_prms = mdl.coefs_ + mdl.intercepts_

    def kerasModel(self):
        # Set-up the network
        #tf.reset_default_graph()
        mdl = Sequential() # model initialization
        for kk in range(self.nb_layer):
            actFun = self.activation if kk<self.nb_layer-1 else self.lastActFun
            mdl.add(Dense(units=self.dims[kk+1], input_dim=self.dims[kk], activation=actFun,\
                          kernel_initializer="random_uniform", bias_initializer="zeros"))
        # Print out the network configuration
        if self.display: print(mdl.summary())
        # Train the network
        if self.opt=='sgd':
            optimizer = SGD(lr=self.LR)
        elif self.opt=='adam':
            optimizer = Adam(lr=self.LR)
        elif self.opt=='rmsprop':
            optimizer = RMSprop(lr=self.LR)
        elif self.opt=='adagrad':
            optimizer = Adagrad(lr=self.LR)
        #optimizer = Adam(lr=self.LR)
        if self.kind == 'regr':
            mdl.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
        elif self.kind == 'binCls':
            mdl.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        elif self.kind == 'multiCls':
            mdl.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        else:
            pass

        YY = self.toOneHotEncoding(self.YY)
        history = mdl.fit(self.XX, YY, epochs=self.nb_epochs, batch_size=self.batchSize, verbose=int(self.display))
        (loss, accuracy) = mdl.evaluate(self.XX, YY, verbose=int(self.display))
        if self.display: print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        if self.kind == 'multiCls':
            Ygrd = np.argmax(mdl.predict(self.XXgrd), axis=1)
        else:
            Ygrd = mdl.predict(self.XXgrd)
        self.nn_Ygrd = Ygrd
        self.lossHistory = history.history['loss']

    def tfModel(self):
        tf.reset_default_graph()
        xx = tf.placeholder(tf.float32, [None, self.dims[0]])
        yy = tf.placeholder(tf.float32, [None, self.dims[-1]])

        prms = {} # model parameters
        intVars = {} # model intermediate variables
        act = xx # network inputs are the previous layer activations to the first layer
        for kk in range(self.nb_layer):
            dIn, dOut = self.dims[kk], self.dims[kk+1]
            prms['W'+str(kk)] = tf.Variable(tf.random_normal([dIn, dOut], stddev=2/(dIn+dOut)), name='W' + str(kk))
            prms['b'+str(kk)] = tf.Variable(tf.zeros([dOut], name='b' + str(kk)))
            act_prev = act
            zz = tf.matmul(act_prev, prms["W"+str(kk)]) + prms["b"+str(kk)]
            actFun = self.activation if kk<self.nb_layer-1 else self.lastActFun
            if actFun == 'relu':
                act = tf.nn.relu(zz)
            elif actFun == 'sigmoid':
                act = tf.nn.sigmoid(zz)
            elif actFun == 'tanh':
                act = tf.nn.tanh(zz)
            elif actFun == 'softmax':
                act = tf.nn.softmax(zz)
            elif actFun == 'linear':
                act = zz
            intVars['z'+str(kk)] = zz
            intVars['a'+str(kk)] = act

        if self.kind == 'regr':
            loss_ = tf.losses.mean_squared_error(labels=yy, predictions=zz)
        elif self.kind == 'binCls':
            loss_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=yy, logits=zz)
        elif self.kind == 'multiCls':
            loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=yy, logits=zz)
        else:
            pass
        self.loss = tf.reduce_mean(loss_, name='loss')

        if self.opt=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.LR, name='sgdOpt')
        elif self.opt=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, name='adamOpt')
        elif self.opt=='rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.LR, name='rmspropOpt')
        elif self.opt=='adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.LR, name='adagradOpt')

        self.optimizer = optimizer.minimize(self.loss)
        if self.kind == 'multiCls':
            self.y_pred = tf.argmax(act, axis=1)
            y_act = tf.argmax(yy, axis=1)
        else:
            self.y_pred = act
            y_act = yy

        if self.kind in ['binCls', 'multiCls']:
            self.correct_pred = tf.equal(tf.round(self.y_pred), y_act, name='correct_pred')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')
        self.xx, self.yy, self.prms, self.intVars = xx, yy, prms, intVars

    def tfNextBatch(self, jj, XX, YY):
        Xb = XX[jj*self.batchSize:(jj+1)*self.batchSize, :]
        Yb = YY[jj*self.batchSize:(jj+1)*self.batchSize, :]
        return Xb, Yb

    def tfTraining(self,):
        # training the tensorflow model
        YY = self.toOneHotEncoding(self.YY)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            lossHistory = []
            for kk in range(self.nb_epochs):
                for jj in range(self.nb_pnt//self.batchSize):
                    Xb, Yb = self.tfNextBatch(jj, self.XX, YY)
                    mdl_loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.xx: Xb, self.yy: Yb})
                lossHistory.append(mdl_loss)
                if kk==self.nb_epochs-1:
                    print('The final model loss is {}'.format(mdl_loss))
            self.lossHistory = np.array(lossHistory)
            self.nn_prms = sess.run(list(self.prms.values()))
            self.nn_vars = sess.run(list(self.intVars.values()), feed_dict={self.xx: self.XX})
            self.nn_W0 = sess.run([self.prms['W0']])
            self.nn_Ygrd = sess.run(self.y_pred, feed_dict={self.xx: self.XXgrd, self.yy: self.toOneHotEncoding(self.YYgrd)})

    def pytorchModel(self):
        # Set-up the network
        mdl = ptFCNN(self.dims, self.activation, self.lastActFun)
        self.mdl = mdl
        # Print out the network configuration
        if self.display: print(list(mdl.parameters()))
        # Train the network
        # loss function
        if self.kind == 'regr':
            lossFun = tnn.MSELoss()
        elif self.kind == 'binCls':
            lossFun = tnn.BCEWithLogitsLoss()
        elif self.kind == 'multiCls':
            #https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/11
            lossFun = tnn.CrossEntropyLoss()
        # optimizer
        if self.opt=='sgd':
            optimizer = torch.optim.SGD(mdl.parameters(), lr=self.LR)
        elif self.opt=='adam':
            optimizer = torch.optim.Adam(mdl.parameters(), lr=self.LR)
        elif self.opt=='rmsprop':
            optimizer = torch.optim.RMSprop(mdl.parameters(), lr=self.LR)
        elif self.opt=='adagrad':
            optimizer = torch.optim.Adagrad(mdl.parameters(), lr=self.LR)

        def ptNextBatch(XX, YY, kind, jj=0, size=None):
            if size:
                XX = XX[jj*size:(jj+1)*size, :]
                YY = YY[jj*size:(jj+1)*size, :]
            Xb = torch.Tensor(XX)
            if kind == 'multiCls':
                Yb = torch.Tensor(YY).long().reshape(-1)
            else:
                Yb = torch.Tensor(YY)
            return Xb, Yb
        # xt = torch.Tensor(self.XX)
        # if self.kind == 'multiCls':
        #     yt = torch.Tensor(self.YY).long().reshape(-1)
        # else:
        #     yt = torch.Tensor(self.YY)
        lossHistory = []
        for epoch in range(self.nb_epochs):
            for jj in range(self.nb_pnt//self.batchSize):
                optimizer.zero_grad()
                # batching
                xtb, ytb = ptNextBatch(self.XX, self.YY, self.kind, jj, self.batchSize)
                # xtb = xt[jj*self.batchSize:(jj+1)*self.batchSize, :]
                # ytb = yt[jj*self.batchSize:(jj+1)*self.batchSize, :]
                # Forward pass: Compute predicted y by passing x to the model
                y_pred, z_pred = mdl(xtb)
                # Compute and print loss
                loss = lossFun(z_pred, ytb)
                loss.backward()
                optimizer.step()
            lossHistory.append(loss.item())

        if self.display: print('The final model loss is {}'.format(loss.item()))
        # accuracy
        xt, yt = ptNextBatch(self.XX, self.YY, self.kind)
        #xt = torch.Tensor(self.XX)
        if self.kind == 'multiCls':
            y_pred = torch.max(mdl(xt)[0].data, 1).indices
            correct = (y_pred == yt).sum().item()
        elif self.kind == 'binCls':
            correct = (torch.round(mdl(xt)[0].data) == yt).sum().item()
        if self.kind in ['binCls', 'multiCls']:
            self.accuracy = correct/xt.shape[0]
            if self.display: print("Final accuracy: {:.4f}%".format(self.accuracy * 100))

        self.lossHistory = np.array(lossHistory)
        self.nn_prms = list(mdl.parameters())


        # grid output
        xtest = torch.Tensor(self.XXgrd)
        y_pred, z_pred = mdl(xtest)
        if self.kind == 'multiCls':
            nn_Ygrd = torch.max(y_pred.data, 1).indices.numpy()
        else:
            nn_Ygrd = y_pred.detach().numpy()
        self.nn_Ygrd = nn_Ygrd

class ptFCNN(tnn.Module):
    #
    def __init__(self, dims, activation, lastActFun):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        """
        super(ptFCNN, self).__init__()
        self.fcs = []
        self.nb_layer = len(dims)-1
        self.dims = dims
        self.activation = activation
        self.lastActFun = lastActFun
        for kk in range(self.nb_layer):
            dIn, dOut = dims[kk], dims[kk+1]
            self.fcs.append(tnn.Linear(dIn, dOut))
        self.fcs = tnn.ModuleList(self.fcs)

    def forward(self, xx):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        act = xx # network inputs are the previous layer activations to the first layer
        for kk in range(self.nb_layer):
            act_prev = act
            zz = self.fcs[kk](act_prev)
            actFun = self.activation if kk<self.nb_layer-1 else self.lastActFun
            if actFun == 'relu':
                act = F.relu(zz)
            elif actFun == 'sigmoid':
                act = torch.sigmoid(zz)
            elif actFun == 'tanh':
                act = torch.tanh(zz)
            elif actFun == 'softmax':
                act = F.softmax(zz, dim=1)
            elif actFun == 'linear':
                act = zz
        return act, zz
