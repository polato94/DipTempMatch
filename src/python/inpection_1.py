import cv2
from matplotlib import pyplot as plt
from termcolor import colored

import imutils as utils
import glob
import initdata as init
import numpy as np
import operator
import numpy.random as rd
import matplotlib.pyplot as plt
import cvHelper as cvh
import time
import types

###########fummelfaktoren###############


generations = 100
matrices_per_generation = 100


fittest_num = int(round( matrices_per_generation/6))
save_fittest = 4

#adjusts the dynamic of the mutations
shrinking_scale_start = 1.5
shrinking_scale = shrinking_scale_start
scalefactor = 1.02
scale_min = 0.5
mutate_percent = 10
muta_factor = 1.01
muta_max = 50
rising_mutating = mutate_percent
##########################  

#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCORR']
#methods = ['cv2.TM_CCOEFF']
#methods = ['cv2.TM_CCORR']



class th_mat_mem:
   score = 0
   th_matrix = []
   matrix_created = False

   def g_matrix(self):
         return np.array([
                            np.random.uniform(-100000, 100000, 8),
                            #np.random.uniform(-1, 1, 8),
                            np.random.uniform(-100000, 100000, 8),
                            np.random.uniform(-1, 1, 8),
                            #np.random.uniform(-1000000000, 1000000000, 8),
                            #np.random.uniform(-1, 1, 8)
                            ])

  

   def adjust_score(self, score):
      self.score = score
      
   def generate_matrix(self):
         self.th_matrix = self.g_matrix()
         self.matrix_created = True

   def generate_with_seed(self, scale, seed):
        add_matrix = self.g_matrix()
        self.th_matrix = seed + add_matrix * scale

   def shuffle_matrix(self,mat_partner,mutate):
     if(not self.matrix_created):
       self.generate_matrix()
     else:
       for i in range(0, len(self.th_matrix)):
         if np.random.randint(0,1) == 0:           
           self.th_matrix[i] = mat_partner[i]
         #else:
         #  self.th_matrix[i] = self.th_matrix[i] * (1 + (np.random.randint(-mutate,mutate)/100))
           

       
   def mutate_matrix(self,mutate):
     if(not self.matrix_created):
       self.generate_matrix()
     else:
       for i in range(0, len(self.th_matrix)):
         self.th_matrix[i] = self.th_matrix[i] * (1 + (np.random.randint(-mutate,mutate)/100))

      
   def adjust_matrix(self, scale):
      #generate adjustment
      if(not self.matrix_created):
        self.generate_matrix()
      else:
        add_matrix = self.g_matrix()
  
        self.th_matrix = self.th_matrix + add_matrix * scale

#0-normal, 1-nohat, 2-noface,3-noleg,4-noBodyprint,5-noHand,6-nohead,7-noarm, 
#matching_thresholds = np.array([0,-15000,-5000,-5000,-5000,-6000,-8000,0])

#threshold ranges found by testing

#    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
 
def get_template_results2(methods,img,defects,imagePath,imgbackground):
  cvh.imgplot(img,imagePath)
  cvh.imgplot(imgbackground,"back")
  
  img_procc = img.copy() 

  
  
  img_procc = utils.shadding(img_procc,imgbackground)
  cvh.imgplot(img_procc,"div")
  cvh.plothist(img_procc,"hist")
  
  img_procc = cv2.cvtColor(img_procc, cv2.COLOR_BGR2GRAY)
  cvh.imgplot(img_procc,"grey")
  
  #img_procc = cvh.gray_spread(img_procc)
  #cvh.imgplot(img_procc,"grey")
  
  #img_procc = utils.imclearborder(img_procc,10)
  #cvh.imgplot(img_procc,"imclearborder")
  
  img_procc = utils.bwareaopen(img_procc,20)
  cvh.imgplot(img_procc,"bwareaopen")
  
  
  return
  
def get_template_results(methods,img, defects):

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
    #slice_am = 25
    #img_gray = img_gray[slice_am:np.shape(img_gray)[0]-slice_am,:]
    #cvh.imgplot(img_gray, "sliced")
    #img_gray = clahe.apply(img_gray)

    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_processed = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel_open, iterations=2)
    #cv2.imshow("opening", img_gray)
    #cv2.waitKey(0)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_processed = cv2.morphologyEx(img_processed, cv2.MORPH_CLOSE, kernel_closing, iterations = 1)
    # Converting image to a binary image 
    # ( black and white only image).
    #cv2.imshow("closing", img_processed)
    #cv2.waitKey(0)
    _, threshold = cv2.threshold(img_processed, 115, 255, cv2.THRESH_BINARY) 
    #cv2.imshow("threshold",threshold)
    
    
    
    
    
    #Detecting contours in image. 
    contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    a = 0;
    phi = 0;
    target=0;
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
        cv2.drawContours(img_gray,[approx], 0, (0,0,255),5)
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
            
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(str(target))
    #print(img_gray.shape)
    if(target != 0):

        phi = target[2]

        center_x = int(target[0][0])
        center_y = int(target[0][1])
        
        #image_cut = utils.rotate_around_point(img,phi,(center_x,center_y))
        #img = img[slice_am:np.shape(img)[0]-slice_am,:]
        image_cut = subimage(img, center=(center_x,center_y), theta = phi, width=int(target[1][0])+35, height=int(target[1][1]+35))
        
        #print(image_cut.shape[0])
        #cvh.showRectangle(img_gray, (center_x,center_y), 10, 10)
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
    #cv2.waitKey()

    #########Start Template Matching
    


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




def inspect_image(methods,img, defects, threshold_matrix):
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
    for it in range(0,8):
       if(voting_array[it] > val):
          val = voting_array[it]
          max_index = it
    
    predicted_label = max_index;
    
    img_processed = img_gray 
    #predicted_label = 0
    return img_processed, predicted_label, template_results



def test_threshold_matrix(methods,defects,threshold_matrix,results):
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
        for it in range(0,8):
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
            print(imagePath)
            img = cv2.imread(imagePath)
            
            if img is None:
               print("Error loading: " + imagePath)
               # end this loop iteration and move on to next image
               continue

            img2 = img.copy()
            #get_template_results2(methods,img2,defects,imagePath, imgbackground)
            images_template_results.append(get_template_results(methods,img,defects))
         dir_images_template_results.append(images_template_results)

start_time = time.time()
rando_group_list = list()

for gen in range(0, generations):

   np.set_printoptions(precision=3) 
  
   #generate and update threashold
   
   if shrinking_scale < scale_min:
      shrinking_scale = scale_min
   else:     
      shrinking_scale = shrinking_scale / scalefactor
      
   if rising_mutating > muta_max:
      rising_mutating = muta_max
   else:     
      rising_mutating = rising_mutating * muta_factor
   
   print()
   print("GENERATION:", gen+1,"/",generations,  "Scale:", shrinking_scale, "Mutate:", rising_mutating)
   for mat in matrix_list:
      #reset variables
      y_true, y_pred = [], []   
      y_pred_old, y_true_old = [],[]
      mat_index = matrix_list.index(mat);
      
      rando_group = 0;

      
      #save first
      #update the best and regenerate the others
      #add 1% random survival
      if gen > 0 and mat_index < save_fittest:
         pass 
        
      elif(mat_index < fittest_num*2):
         mat.shuffle_matrix(matrix_list[np.random.randint(0,fittest_num-1)].th_matrix,
                            rising_mutating)
         mat.mutate_matrix(rising_mutating)
         rando_group = 1
         
      #elif(mat_index < fittest_num*3):
      #   mat.th_matrix = matrix_list[0].th_matrix.copy()
      #   mat.mutate_matrix(rising_mutating)
      #   rando_group = 3
         
      elif(np.random.randint(1,50) == 25):
        mat.mutate_matrix(rising_mutating)
        rando_group = 2
        
      elif(np.random.randint(0,1) == 1):
        mat.generate_matrix()
        mat.shuffle_matrix(matrix_list[np.random.randint(0,save_fittest-1)].th_matrix,
                            rising_mutating) 
        rando_group = 3
        
      else:
        mat.generate_matrix()
        rando_group = 4
   
      
    
      dir_count = 0;
      for class_label, defect_type in enumerate(defects):
          img_count = 0;
          single_folder_img_results = dir_images_template_results[dir_count]
          for img_results in single_folder_img_results:

              predicted_label = test_threshold_matrix(methods,defects, mat.th_matrix, img_results)
              
              
              y_pred.append(predicted_label)
              y_true.append(class_label+1)
              img_count = img_count + 0;
          dir_count = dir_count +1;
      

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
         shrinking_scale = shrinking_scale_start
         rising_mutating = mutate_percent
         if(gen!=0):
           rando_group_list.append(rando_group)
         print(colored("better matrix found: ", 'red'), best_acc, "group:",rando_group)
         #print(bcolors.WARNING+"better matrix found: "+ bcolors.ENDC, best_acc, "group:",rando_group)
         
        # best_mat = thr_matrix

   #sort list by score
   matrix_list.sort(key=operator.attrgetter('score'), reverse=True)
   
   avg_acc_fittest = 0;
   #print scores:
   best_acc_list.append(matrix_list[0].score);
   for best in range(fittest_num):
     #print("score", matrix_list[best].score)
     avg_acc_fittest = avg_acc_fittest + matrix_list[best].score     
   avg_acc_fittest = avg_acc_fittest/fittest_num
   print("top score:",matrix_list[0].score)
   print("avg score top", fittest_num ,":", avg_acc_fittest)
    
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
print("best from group",rando_group_list)
print("best pred:",best_y_pred)
print("true:     ",y_true)
print(confusion_matrix(y_true, best_y_pred))

print("best 3 are")
for win in range(3):
   filename = "winner_place"+str(win)+"("+str(matrix_list[win].score)+".csv";
   np.savetxt(filename,matrix_list[win].th_matrix, delimiter=",")
   print("#", win, " with score:  ", matrix_list[win].score, "is:")
   print(matrix_list[win].th_matrix)
   


