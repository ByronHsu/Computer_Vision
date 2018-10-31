def draw_lda_test(ldas, mean):
   img_list = []
   o_prefix = 'p2_output/'
   i_prefix = 'p2_data/'
   for i in class_list:
      for j in test_list:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(i_prefix + file_name, 0)
         img = img.flatten()
         img_list.append(img)
   imgs = np.array(img_list, dtype = float)

   # ldas: 2576 x A, imgs: 1 x 2576 
   Y = (imgs - mean) @ ldas # N x A, 一個row代表一個投影
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em, alpha = 0.8, edgecolors='none')
   plt.title('LDA Scattering DIM = 100')
   plt.savefig(o_prefix + 'LDA-scattering.png')
   print('Saving LDA-scattering.png ...')