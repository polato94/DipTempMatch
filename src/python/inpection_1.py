import cv2
from matplotlib import pyplot as plt

import imutils as utils
import glob
import initdata as init
import numpy as np
import operator
import numpy.random as rd
import matplotlib.pyplot as plt
import time

###########fummelfaktoren###############


generations = 10
matrices_per_generation = 350

fittest_num = int(round( matrices_per_generation/8))


#adjusts the dynamic of the mutations
shrinking_scale = 1.5
scalefactor = 1.01

##########################  


class th_mat_mem:
   score = 0
   th_matrix = []
   matrix_created = False

  

   def adjust_score(self, score):
      self.score = score
      
   def generate_matrix(self):
         self.th_matrix = np.array([
                            np.random.uniform(-100000, 100000, 8),
                            np.random.uniform(-1, 1, 8),
                            np.random.uniform(-100000, 100000, 8),
                            np.random.uniform(-1, 1, 8),
                            np.random.uniform(-1000000000, 1000000000, 8),
                            np.random.uniform(-1, 1, 8)])
         self.matrix_created = True

   def generate_with_seed(self, scale, seed):
        add_matrix = np.array([
                              np.random.uniform(-100000, 100000, 8),
                              np.random.uniform(-1, 1, 8),
                              np.random.uniform(-100000, 100000, 8),
                              np.random.uniform(-1, 1, 8),
                              np.random.uniform(-1000000000, 1000000000, 8),
                              np.random.uniform(-1, 1, 8)])
        self.th_matrix = seed + add_matrix * scale

      
      
   def adjust_matrix(self, scale):
      #generate adjustment
      if(not self.matrix_created):
        self.generate_matrix()
      else:
        add_matrix = np.array([
                              np.random.uniform(-100000, 100000, 8),
                              np.random.uniform(-1, 1, 8),
                              np.random.uniform(-100000, 100000, 8),
                              np.random.uniform(-1, 1, 8),
                              np.random.uniform(-1000000000, 1000000000, 8),
                              np.random.uniform(-1, 1, 8)])
  
        self.th_matrix = self.th_matrix + add_matrix * scale

#0-normal, 1-nohat, 2-noface,3-noleg,4-noBodyprint,5-noHand,6-nohead,7-noarm, 
#matching_thresholds = np.array([0,-15000,-5000,-5000,-5000,-6000,-8000,0])

#threshold ranges found by testing

#    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']



def subimage(image, center, theta, width, height):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''
   #print("center = ")
   #print(center)
   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image
 
def get_template_results(img, defects):

    #print("Image shape = ",img.shape)
    bg_b,bg_g,bg_r = cv2.split(imgbackground)
    bg_r = bg_r + 0.0001
    bg_b = bg_b + 0.0001
    bg_g = bg_g + 0.0001
    b, g, r = cv2.split(img)
    
    b = b / bg_b;
    g = g / bg_g;
    r = r / bg_r;

    #print(b)
    
    b = ((b / b.max()) * 255).astype(np.uint8)
    #b.max();
    #print(b)
    g = ((g / g.max()) * 255).astype(np.uint8)
    r = ((r / r.max()) * 255).astype(np.uint8)
    
    img_processed = cv2.merge([r, r, r])

    #cv2.imshow("backgcomp", img_processed)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize = (8,8))

    #convert to gray
    img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)

    #img_gray = clahe.apply(img_gray)

    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_processed = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_open, iterations=2)
    #cv2.imshow("opening", img_gray)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, kernel_closing, iterations = 1)
    # Converting image to a binary image 
    # ( black and white only image).
    #cv2.imshow("closing", img_processed)
    _, threshold = cv2.threshold(img_processed, 115, 255, cv2.THRESH_BINARY) 
    #cv2.imshow("threshold",threshold)
    #Detecting contours in image. 
    contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    a = 0;
    phi = 0;
    target=0;
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
        #cv2.drawContours(img_gray,[approx], 0, (0,0,255),5)
        #draw rectangle around contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_gray, [box], 0, (0,0,255),2)
        a_n = rect[1][0]*rect[1][1]
        #print(str(rect))
        if ((a_n > a) and (rect[1][0] < 280)):
            a = a_n
            target = rect

    #print(str(target))
    #print(img_gray.shape)
    if(target != 0):

        phi = target[2]

        center_x = int(target[0][0])
        center_y = int(target[0][1])
        
        image_cut = subimage(img, center=(center_x,center_y), theta = phi, width=int(target[1][0])+35, height=int(target[1][1]+35))
        #print(image_cut.shape[0])
        if((image_cut.shape[0] == 0) or (image_cut.shape[1] == 0)):
            print("could not cut image")
            return img_gray, 0;
        
        img_processed = image_cut;
    else:
        print("no target detected")
        img_processed = img_gray
        return img_gray, 0
        
    #print(image_cut.shape)

    # rotate cut image to vertical
    #print(image_cut)
    if(image_cut.shape[0] < image_cut.shape[1]):
       image_cut = cv2.rotate(image_cut, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #cv2.imshow("cut image",image_cut)


    #########Start Template Matching
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


    #print("cut image shape --before interpol", image_cut.shape)
    #print("defect template --before interpol ",defects['hat']['mask'].shape)

    img_cut_gray = cv2.cvtColor(image_cut,cv2.COLOR_BGR2GRAY)

    #interpolate smaller image with zeros -- only to be done once bc all templates are 124x200
    if(image_cut.shape[1] < defects['hat']['mask'].shape[1]):
       diff = defects['hand']['mask'].shape[1]-image_cut.shape[1]
       #print("image not wide enough by:", diff)
       img_cut_gray = cv2.copyMakeBorder(img_cut_gray,
                                      0,
                                      0,
                                      diff,
                                      diff,
                                      borderType = cv2.BORDER_CONSTANT,
                                      value = [0,0,0])
       
   
    if(image_cut.shape[0] < defects['hat']['mask'].shape[0]):
       diff = defects['hand']['mask'].shape[0]-image_cut.shape[0]
       #print("image not high enough by: ",diff)
       img_cut_gray = cv2.copyMakeBorder(img_cut_gray,
                                      diff,
                                      diff,
                                      0,
                                      0,
                                      borderType = cv2.BORDER_CONSTANT,
                                      value = [0,0,0])
    img_match = img_cut_gray.copy()
    
    template_results = list();
    #todo: wade trough matching methods and create voting map-problem: dialing in Thresholds!!
    #wade trough templates and check thresholds
    meth_iter = 0;
    for meth in methods:
       temp_iter = 0;
       for temps in defects:
          temp_iter = temp_iter + 1;
          #print(temps)
          #print("cut image shape", img_match.shape)
          #print("defect template= ", defects[temps]['mask'].shape)
          res = cv2.matchTemplate(img_match, defects[temps]['mask'], eval(meth))
          template_results.append(res)          
       meth_iter = meth_iter +1;
             
    return template_results

def inspect_image(img, defects, threshold_matrix):
    voting_array =np.array( [0,0,0,0,0,0,0,0])
    #print("Image shape = ",img.shape)
    bg_b,bg_g,bg_r = cv2.split(imgbackground)
    bg_r = bg_r + 0.0001
    bg_b = bg_b + 0.0001
    bg_g = bg_g + 0.0001
    b, g, r = cv2.split(img)
    
    b = b / bg_b;
    g = g / bg_g;
    r = r / bg_r;

    #print(b)
    
    b = ((b / b.max()) * 255).astype(np.uint8)
    #b.max();
    #print(b)
    g = ((g / g.max()) * 255).astype(np.uint8)
    r = ((r / r.max()) * 255).astype(np.uint8)
    
    img_processed = cv2.merge([r, r, r])

    #cv2.imshow("backgcomp", img_processed)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize = (8,8))

    #convert to gray
    img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)

    #img_gray = clahe.apply(img_gray)

    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_processed = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_open, iterations=2)
    #cv2.imshow("opening", img_gray)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, kernel_closing, iterations = 1)
    # Converting image to a binary image 
    # ( black and white only image).
    #cv2.imshow("closing", img_processed)
    _, threshold = cv2.threshold(img_processed, 115, 255, cv2.THRESH_BINARY) 
    #cv2.imshow("threshold",threshold)
    #Detecting contours in image. 
    contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    a = 0;
    phi = 0;
    target=0;
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
        #cv2.drawContours(img_gray,[approx], 0, (0,0,255),5)
        #draw rectangle around contour
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_gray, [box], 0, (0,0,255),2)
        a_n = rect[1][0]*rect[1][1]
        #print(str(rect))
        if ((a_n > a) and (rect[1][0] < 280)):
            a = a_n
            target = rect

    #print(str(target))
    #print(img_gray.shape)
    if(target != 0):

        phi = target[2]

        center_x = int(target[0][0])
        center_y = int(target[0][1])
        
        image_cut = subimage(img, center=(center_x,center_y), theta = phi, width=int(target[1][0])+35, height=int(target[1][1]+35))
        #print(image_cut.shape[0])
        if((image_cut.shape[0] == 0) or (image_cut.shape[1] == 0)):
            print("could not cut image")
            return img_gray, 0;
        
        img_processed = image_cut;
    else:
        print("no target detected")
        img_processed = img_gray
        return img_gray, 0
        
    #print(image_cut.shape)

    # rotate cut image to vertical
    #print(image_cut)
    if(image_cut.shape[0] < image_cut.shape[1]):
       image_cut = cv2.rotate(image_cut, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #cv2.imshow("cut image",image_cut)


    #########Start Template Matching
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


    #print("cut image shape --before interpol", image_cut.shape)
    #print("defect template --before interpol ",defects['hat']['mask'].shape)

    img_cut_gray = cv2.cvtColor(image_cut,cv2.COLOR_BGR2GRAY)

    #interpolate smaller image with zeros -- only to be done once bc all templates are 124x200
    if(image_cut.shape[1] < defects['hat']['mask'].shape[1]):
       diff = defects['hand']['mask'].shape[1]-image_cut.shape[1]
       #print("image not wide enough by:", diff)
       img_cut_gray = cv2.copyMakeBorder(img_cut_gray,
                                      0,
                                      0,
                                      diff,
                                      diff,
                                      borderType = cv2.BORDER_CONSTANT,
                                      value = [0,0,0])
       
   
    if(image_cut.shape[0] < defects['hat']['mask'].shape[0]):
       diff = defects['hand']['mask'].shape[0]-image_cut.shape[0]
       #print("image not high enough by: ",diff)
       img_cut_gray = cv2.copyMakeBorder(img_cut_gray,
                                      diff,
                                      diff,
                                      0,
                                      0,
                                      borderType = cv2.BORDER_CONSTANT,
                                      value = [0,0,0])
    img_match = img_cut_gray.copy()
    
    template_results = list();
    #todo: wade trough matching methods and create voting map-problem: dialing in Thresholds!!
    #wade trough templates and check thresholds
    meth_iter = 0;
    for meth in methods:
       temp_iter = 0;
       for temps in defects:
          temp_iter = temp_iter + 1;
          #print(temps)
          #print("cut image shape", img_match.shape)
          #print("defect template= ", defects[temps]['mask'].shape)
          res = cv2.matchTemplate(img_match, defects[temps]['mask'], eval(meth))
          template_results.append(res)
          #print("template match gave= ", res)
          min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
          w, h = defects['hat']['mask'].shape[::-1]
          #print("minimum",min_val, max_val, min_loc, max_loc)
          top_left = min_loc;
          bottom_right = (top_left[0] + w,top_left[1] + h)
          cv2.rectangle(img_match,top_left, bottom_right, 255, 2)
          #cv2.imshow("matched",res)
          if(np.abs(min_val) > threshold_matrix[meth_iter][temp_iter]):
             voting_array[temp_iter] = voting_array[temp_iter]+1
       meth_iter = meth_iter +1;
             
    ########
    #print("voting array is:",voting_array);
    #find most vote
    max_index = 0;
    val = voting_array[0]
    for it in range(0,7):
       if(voting_array[it] > val):
          val = voting_array[it]
          max_index = it
    
    predicted_label = max_index;
    
    img_processed = img_gray 
    #predicted_label = 0
    return img_processed, predicted_label, template_results



def test_threshold_matrix(defects,threshold_matrix,results):
    import types
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    voting_array =np.array( [0,0,0,0,0,0,0,0])
    
    #todo results empty return nothing
    if type(results) is tuple:
      return 0
    
    counter_iter = 0;
    meth_iter = 0;
    for meth in methods:
        temp_iter = 0;
        for temps in defects:
           temp_iter = temp_iter + 1;
           
           min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results[counter_iter])
           #cv2.imshow("matched",res)
           if(np.abs(min_val) > threshold_matrix[meth_iter][temp_iter]):
              voting_array[temp_iter] = voting_array[temp_iter]+1
           counter_iter = counter_iter+1;
        meth_iter = meth_iter +1;
        
        
        ########
        #print("voting array is:",voting_array);
        #find most vote
        max_index = 0;
        val = voting_array[0]
        for it in range(0,7):
          if(voting_array[it] > val):
             val = voting_array[it]
             max_index = it
        
        predicted_label = max_index;

        #predicted_label = 0
    
    return predicted_label

# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

template, defects = init.initdata()

do_plot = False # Enable plotting of images

best_acc = 0
avg_score = 0

best_mat = np.zeros((6,8))

#initialize matrix list

matrix_list = [th_mat_mem() for mat_mem_i in range(matrices_per_generation)]


avg_acc_fittest_list = list();
avg_acc_fittest_list.append(0);
best_acc_list = list();
best_acc_list.append(0);
best_y_pred = list();

# get template matching results from images
dir_images_template_results = list();
for class_label, defect_type in enumerate(defects):


         imageDir = "../../img/" + defects[defect_type]['dir']

         # read all images from folders given in a list
         images_template_results = list();
         for imagePath in glob.glob(imageDir + "*.jpg"):

            img = cv2.imread(imagePath)
            if img is None:
               print("Error loading: " + imagePath)
               # end this loop iteration and move on to next image
               continue


            images_template_results.append(get_template_results(img,defects))
         dir_images_template_results.append(images_template_results)

start_time = time.time()

for gen in range(0, generations):

   
   #generate and update threashold
   shrinking_scale = shrinking_scale / scalefactor
   print()
   print("GENERATION:", gen+1,"/",generations,  "Scale:", shrinking_scale)
   for mat in matrix_list:
      #reset variables
      y_true, y_pred = [], []   
      y_pred_old, y_true_old = [],[]
      mat_index = matrix_list.index(mat);

      
      #save first
      #update the best and regenerate the others
      #add 1% random survival
      if gen > 0 and mat_index == 0:
          pass
      elif(mat_index < fittest_num):
         mat.adjust_matrix(shrinking_scale)
      elif(mat_index < fittest_num*2):
         mat.generate_with_seed(shrinking_scale,matrix_list[0].th_matrix)
      elif(np.random.randint(1,50) == 25):
        mat.adjust_matrix(shrinking_scale)
      else:
        mat.generate_matrix()
   
      
    
      dir_count = 0;
      for class_label, defect_type in enumerate(defects):
          img_count = 0;
          single_folder_img_results = dir_images_template_results[dir_count]
          for img_results in single_folder_img_results:

              predicted_label = test_threshold_matrix(defects, mat.th_matrix, img_results)
              
              
              y_pred.append(predicted_label)
              y_true.append(class_label+1)
              img_count = img_count + 0;
          dir_count = dir_count +1;
      """
      
      for class_label, defect_type in enumerate(defects):


         imageDir = "../../img/" + defects[defect_type]['dir']

         # read all images from folders given in a list

         for imagePath in glob.glob(imageDir + "*.jpg"):

            img = cv2.imread(imagePath)
            if img is None:
               print("Error loading: " + imagePath)
               # end this loop iteration and move on to next image
               continue


            img_processed, predicted_label = inspect_image(img, defects, mat.th_matrix)
            #         img_processed = img
            #         predicted_label = 0
            
            y_pred_old.append(predicted_label)
            y_true_old.append(class_label)  # append real class label to true y's
         
            if (do_plot):
               f, (ax1, ax2) = plt.subplots(1, 2)#
               ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
               ax1.axis("off")
               ax1.set_title(imagePath)
               ax2.imshow(img_processed, cmap='gray')
               ax2.axis("off")
               ax2.set_title("Processed image")
               plt.show()


               #from sklearn.metrics import accuracy_
      #cv2.waitKey(1000)
              """

      from sklearn.metrics import accuracy_score, confusion_matrix
      #print("Matrix:", mat_index+1, '/',matrices_per_generation," Gen:", gen+1, '/', generations ,"Accuracy: ", accuracy_score(y_true, y_pred))
      #pint("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

      #check how good matrix is:
      ma_score = accuracy_score(y_true, y_pred)
      mat.adjust_score(ma_score)
      
      if(best_acc < ma_score):
         # current matrix is better
         best_acc = ma_score
         best_y_pred = y_pred
         print("better matrix found: ", best_acc)
         
        # best_mat = thr_matrix

   #sort list by score
   matrix_list.sort(key=operator.attrgetter('score'), reverse=True)
   
   avg_acc_fittest = 0;
   #print scores:
   best_acc_list.append(matrix_list[0].score);
   for best in range(fittest_num):
     print("score", matrix_list[best].score)
     avg_acc_fittest = avg_acc_fittest + matrix_list[best].score
   avg_acc_fittest = avg_acc_fittest/fittest_num
   print("avg", avg_acc_fittest)
    
   avg_acc_fittest_list.append(avg_acc_fittest)
   
   
   plt.plot(best_acc_list, color='r') 
   plt.plot(avg_acc_fittest_list,color="b")
   plt.ylim(0,1)
   plt.grid()
   plt.show()
    
   #for m_score in matrix_list:
   #  print("score", m_score.score)

print()
print("--- %s seconds ---" % (time.time() - start_time))

print("best 3 are")
for win in range(3):
   filename = "winner_place"+str(win)+"("+str(matrix_list[win].score)+".csv";
   np.savetxt(filename,matrix_list[win].th_matrix, delimiter=",")
   print("#", win, " with score:  ", matrix_list[win].score, "is:")
   print(matrix_list[win].th_matrix)
   


