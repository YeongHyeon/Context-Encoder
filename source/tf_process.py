import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from random import *
from PIL import Image
from sklearn.decomposition import PCA

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data, height, width):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+height, (x*dw):(x*dw)+width, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, height, width, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i], height=height, width=width))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()

def generate_random_mask(x, height, width):

    h_s, w_s = randrange(height//2), randrange(width//2)
    h_d, w_d = randint(height//4, height//2), randint(width//4, width//2)
    h_e, w_e = h_s + h_d, w_s + w_d
    m = np.zeros_like(x)
    m[:, h_s:h_e, w_s:w_e, :] = 1

    return m

def generate_static_mask(x, height, width):

    h_d, w_d = height//3, width//3
    h_s, w_s = height//2 - h_d//2, width//2 - w_d//2
    h_e, w_e = h_s + h_d, w_s + w_d
    m = np.zeros_like(x)
    m[:, h_s:h_e, w_s:w_e, :] = 1

    return m

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="training")

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 10
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, _ = dataset.next_train(batch_size=test_size, fix=True)
        m_tr = generate_static_mask(x=x_tr, height=dataset.height, width=dataset.width)

        x_masked, x_restore = sess.run([neuralnet.drop, neuralnet.x_hat], \
            feed_dict={neuralnet.x:x_tr, neuralnet.m:m_tr, neuralnet.batch_size:x_tr.shape[0]})

        save_img(contents=[x_tr, x_masked, x_masked + (x_restore * m_tr), (x_tr-x_restore)**2], \
            height=dataset.height, width=dataset.width, \
            names=["Input\n(x)", "Masked\n(from x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("training", "%08d.png" %(epoch)))

        while(True):
            x_tr, terminator = dataset.next_train(batch_size)
            m_tr = generate_random_mask(x=x_tr, height=dataset.height, width=dataset.width)

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr, neuralnet.m:m_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)
            loss_rec, loss_adv, loss_tot = sess.run([neuralnet.loss_rec, neuralnet.loss_adv, neuralnet.loss_tot], \
                feed_dict={neuralnet.x:x_tr, neuralnet.m:m_tr, neuralnet.batch_size:x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Rec:%.3f, Adv:%.3f, Tot:%.3f" \
            %(epoch, epochs, iteration, loss_rec, loss_adv, loss_tot))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    make_dir(path="test")

    while(True):
        x_te, terminator = dataset.next_test(1)
        m_te = generate_static_mask(x=x_te, height=dataset.height, width=dataset.width)

        x_masked, x_restore, restore_loss = sess.run([neuralnet.drop, neuralnet.x_hat, neuralnet.mse_r], \
            feed_dict={neuralnet.x:x_te, neuralnet.m:m_te, neuralnet.batch_size:x_te.shape[0]})

        [h, w, c] = x_te[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_masked[0]
        canvas[:, w*2:, :] = x_masked[0] + (x_restore[0] * m_te[0])

        result = Image.fromarray((canvas * 255).astype(np.uint8))
        result.save(os.path.join("test", "%08d.png" %(dataset.idx_te)))

        if(terminator): break
