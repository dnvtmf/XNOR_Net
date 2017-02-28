import matplotlib.pyplot as plt

from bin_cnn import CNN
from data_utils import *
from solver import Solver

data = get_mnist_data()
# data = get_CIFAR10_data()
# for idx in xrange(25):
#     plt.subplot(5, 5, idx + 1)
#     plt.imshow(data['X_train'][idx, 0], cmap='gray')
# plt.show()

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
for k, v in data.iteritems():
    print '%s: ' % k, v.shape


def conv(num_filters, filter_size):
    return {'name': 'conv', 'num_filters': num_filters, 'filter_size': filter_size}


def fc(hidden_dim):
    return {'name': 'fc', 'hidden_dim': hidden_dim}


def pool():
    return {'name': 'pool'}


def norm():
    return {'name': 'norm'}


def flat():
    return {'name': 'flat'}


input_dim = X_train.shape[1:]
arch = [conv(32, (5, 5)), norm(), flat(), fc(100), norm(), fc(100), norm(), fc(10)]
model = CNN(arch, input_dim=input_dim, weight_scale=3e-2, reg=1e-2)
solver = Solver(model, data,
                num_epochs=10, batch_size=200,
                update_rule='adam',
                optim_config={
                    'learning_rate': 3e-4
                },
                print_every=100,
                verbose=True)
solver.train()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.show()

y_test_pred = np.argmax(model.loss(X_test), axis=1)
y_val_pred = np.argmax(model.loss(X_val), axis=1)
print 'Validation set accuracy: ', (y_val_pred == y_val).mean()
print 'Test set accuracy: ', (y_test_pred == y_test).mean()
