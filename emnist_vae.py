import graphlearning as gl



data, labels = gl.datasets.load("emnist", metric="raw")

print("Training VAE....")
data_vae = gl.weightmatrix.vae(data, epochs=1000)
gl.datasets.save(data_vae, labels, "emnist", metric="vae", overwrite=False)

print("Constructing similarity graphs")
W_raw = gl.weightmatrix.knn(data, 20)
W_vae = gl.weightmatrix.knn(data_vae, 20)

G_raw = gl.graph(W_raw)
print(f"raw graph connected = {G_raw.isconnected()}")
G_vae = gl.graph(W_vae)
print(f"vae graph connected = {G_vae.isconnected()}")

num_train_per_class = 1
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

pred_labels_raw = gl.ssl.poisson(W_raw).fit_predict(train_ind,train_labels)
pred_labels_vae = gl.ssl.poisson(W_vae).fit_predict(train_ind,train_labels)

accuracy_raw = gl.ssl.ssl_accuracy(labels,pred_labels_raw,len(train_ind))
accuracy_vae = gl.ssl.ssl_accuracy(labels,pred_labels_vae,len(train_ind))

print('Raw Accuracy: %.2f%%'%accuracy_raw)
print('VAE Accuracy: %.2f%%'%accuracy_vae)
