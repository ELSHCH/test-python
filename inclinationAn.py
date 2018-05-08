''' ***********************************************************************************************************************************
*
* This code implements the training step of recession algorithm using recession data from the past with known period of recession.
*
* Last modified by E. Shchekinova 2017
"
/* Definition of fixed global parameters *********************************************************************************************'''
  import math
  import sys
  import random
 
  sizeM = 3 # length of fixed running window used for selecting from the time series, two options could be selected  sizeM = {2 , 3} */
  numberVar = 2 # number of time series used for training, minimum permitted is  numberVar = 1 and maximum permitted  numberVar =3 */
  max_int = 10 # maximal size of preselected sequences in optimization*/
  maxLen = 400    # string length */
  flag_classes = 1 # if  flag_classes = 1 instead of full table of binary codes the grouping into classes is used */
  trans_total = 0 ''' 0 is used when probability of transition is calculated for every recession separately,
                         1  is used when probability of transition is defined from the total time series '''

  q = [0.0, 0.3, 0.5, 1.0] # q is the parameter that defines probability of "+" to occur in a binary sequence of size T*/
  size_q = q.size()
  t = [4, 6, 8]  # T is a prediction base*/
  size_T =t.size() 
  eps = [1.0,10.0,100.0,200.0] # eps is a threshold of relative change */
  size_eps = eps.size() 
  decay_kernel=[5,10]  # decay_kernel used for weighting functions*/
  size_dk = decay_kernel.size()
  prob_total = [
                       [0.9540, 0.0, 0.0, 0.0, 0.8636, 0.0, 0.0, 0.0],
                       [0.0460, 0.0, 0.0, 0.0, 0.1364, 0.0, 0.0, 0.0],
                       [0.0,    0.0, 0.0, 0.0, 0.0,    0.0, 0.0, 0.0],
                       [0.0,    1.0, 0.0, 0.0, 0.0,    1.0, 0.0, 0.0],
                       [0.0,    0.0, 0.5, 0.0, 0.0,    0.0, 0.8800, 0.0],
                       [0.0,    0.0, 0.5, 0.0, 0.0,    0.0, 0.12, 0.0],
                       [0.0,    0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4651],
                       [0.0,    0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.5349]
                        ]  ''' probability transition matrix estimated
                                                               the total time series of recession,
                                                               it is used when TRANS_TOTAL=1 '''
  amb = [2, 4]  # amb is a prediction ambition parameter*/
  size_amb = amb.size() # size_amb is a size of array of ambition parameter*/
'''  length_recession - is known length of recession in the past used for optimization (given in months) 
 *   length_pre_recession - is the length of pre-recession period 
 *   length_post_recession - is the length of post-recession period 
 *   mid_point - is the start of pre-recession 
 *   start_recession = mid_point + length_pre_recession - is the beginning of recession 
 *   length_data - is the length of the entire recession time series 
 *   total_length = length_pre_recession + length_recession + length_post_recession - is the total period of recession including pre- and post- recessions 
 
     start_current_period - beginning of the period from which a prediction is made (number of month) 
 *    end_current_period - end of the period from which a prediction is made 
 *    start_predicted_period - end of the prediction period  
 *
 *
 '''
  def calculateSTD(length_data, data[length_data][ numberVar], stdD[ numberVar]):

    for j in range(numberVar):
       sum = 0.0 
       standardDeviation=0.0 
    for i in range(length_data):
        sum += data[i][j]

    mean = sum/length_data 

    for i in range(length_data):
        standardDeviation = standardDeviation+ math.pow(data[i][j] - mean, 2) 
        stdD[j] = math.sqrt(standardDeviation/(length_data-1))
    
  def myRandom(size_n): 
# Function returns an array of integer random numbers within range [1 size_n]
  numNums = 0
 
    # Initialize with a specific size.

    if (size_n >= 0): 
       
        for i in range(size_n):
         numArr[i] = i
         numNums = size_n

    if (numNums == 0):
       return err_no_num

    '''' Get random number from pool and remove it (rnd in this
       case returns a number between 0 and numNums-1 inclusive).''''

    n = random.random() % numNums 
    i = numArr[n] 
    numArr[n] = numArr[numNums-1] 
    numNums = numNums - 1 
    if (numNums == 0): 
        numArr = 0

  return i

def GridNew(length_data, f[length_data][ numberVar], coarse_f[length_data- sizeM][ sizeM][ numberVar], thres_var, class_f[length_data -  sizeM][ numberVar], *var_th):
''' This function is used to replace an original time series with the sequence of ("1", "-1", "0") according to defined rules
*  and given step-to-step variation threshold and finally with a class number assigned to every fixed time window  sizeM from a sequence
*
*  thres_var - variation threshold
*  f[length_data][ numberVar] - is an original time series
*  length_data - is the length of time series,  numberVar is the number of variables used in the analysis (default  numberVar = 2)
*  coarse_f[n- sizeM][ sizeM][ numberVar] - is resulting sequence of ("1", "-1", "0")
*  class_f[length_data -  sizeM][ numberVar] - array of class numbers
'''
   name_file_cgn = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/cgn.dat" 
 # open file for writing binary table F */

   file_cgn = open(name_file_cgn,"w") 

   for i in range(numberVar):
     max_f[i] = 0.0 
     for k in range(length_data):
       if ((f[k][i] > max_f[i])&&(f[k][i]>0)):
          max_f[i] = f[k][i] 
       if ((-f[k][i] > max_f[i])&&(f[k][i]<0)): 
          max_f[i] = -f[k][i] 
 
   #normalization of time series

   calculateSTD(length_data,f,stdD) 
   for i in range(numberVar):
     for k in range(length_data):
       if (max_f[i]>0): 
          norm_f[k][i] = f[k][i]/max_f[i] 
   '''         if (stdD[i]>0):
          norm_f[k][i] = f[k][i]/stdD[i] '''
        
     MinVarFunc(length_data, norm_f, &thres) 

   # coarse graining of time series
     for i in range(length_data - sizeM):
     for j in range( sizeM):
       for k in range(numberVar):
        if (norm_f[j+i][k] == 0):
          coarse_f[i][j][k]=0 
          n_f[j+i+1][k]=norm_f[j+i][k] 
        else:
     if ((fabs(norm_f[j+i+1][k]-norm_f[j+i][k]) > thres*thres_var)&&(norm_f[j+i+1][k]-norm_f[j+i][k]>0)):
          #  printf("%g\t%g\t%g\n",norm_f[j+i+1][k]-norm_f[j+i][k],thres_var*thres,thres) 
           coarse_f[i][j][k] = 1 
           n_f[j+i+1][k]=norm_f[j+i][k] 
     else if: ((fabs(norm_f[j+i+1][k]-norm_f[j+i][k]) >  thres*thres_var)&&(norm_f[j+i+1][k]-norm_f[j+i][k]<0)):
        # printf("%g\t%g\n",norm_f[j+i+1][k]-norm_f[j+i][k],thres_var*thres) 
           coarse_f[i][j][k] = -1 
            n_f[j+i+1][k]=norm_f[j+i][k] 
     else:
        #        printf("%g\t%g\t%g\n",norm_f[j+i+1][k]-norm_f[j+i][k],thres_var*thres,thres) 
           coarse_f[i][j][k] = 0 
            n_f[j+i+1][k]=norm_f[j+i][k] 
      
     *var_th=thres*thres_var 
     for k in range(numberVar):
     for i in range(length_data-1):
       file_cgn.write(norm_f[i][k]+'\t') 

     file_cgn.write("%g\n",norm_f[length_data-1][k]) 
     file_cgn.close() 


     if ( flag_classes == 1):
     for k in range( numberVar):
        for i in range(length_data -  sizeM):
           n_u = 0 
           n_d = 0 
           n_n = 0 
           for j in range(sizeM):
              if (coarse_f[i][j][k]==1):
                n_u = n_u + 1 
              
              if (coarse_f[i][j][k]==0):
                n_n = n_n + 1 
              
              if (coarse_f[i][j][k]==-1):
                n_d = n_d + 1 
              
              n_und_f[i][0] = n_u 
              n_und_f[i][1] = n_n 
              n_und_f[i][2] = n_d 
      
        if (sizeM == 3):
          for i in range(length_data -  sizeM):
            if (n_und_f[i][0]==3):
               class_f[i][k]=1 
            
            if ((n_und_f[i][0]==2)&&(n_und_f[i][1]==1)):
               class_f[i][k]=1 
            
            if ((n_und_f[i][0]==1)&&(n_und_f[i][1]==2)):
               class_f[i][k]=2 
            
            if ((n_und_f[i][0]==2)&&(n_und_f[i][2]==1)):
               class_f[i][k]=2 
            
            if ((n_und_f[i][2]==1)&&(n_und_f[i][1]==1)&&(n_und_f[i][0]==1)):
               class_f[i][k]=3 
            
            if (n_und_f[i][1]==3){
               class_f[i][k]=3 
            
            if ((n_und_f[i][1]==2)&&(n_und_f[i][2]==1)):
               class_f[i][k]=4 
            
            if ((n_und_f[i][0]==1)&&(n_und_f[i][2]==2)):
               class_f[i][k]=4 
            
            if (n_und_f[i][2]==3):
               class_f[i][k]=5 
            
            if ((n_und_f[i][1]==1)&&(n_und_f[i][2]==2)):
               class_f[i][k]=5 
           
     if ( sizeM == 2):
      for i in range(length_data -  sizeM):
         if (n_und_f[i][0]==2):
            class_f[i][k]=1 
         
         if (n_und_f[i][2]==2):
            class_f[i][k]=3 
         
         if ((n_und_f[i][0]==1)&&(n_und_f[i][1]==1)):
            class_f[i][k]=1 
         
         if ((n_und_f[i][2]==1)&&(n_und_f[i][1]==1)):
            class_f[i][k]=3 
         
         if (n_und_f[i][1]==2):
            class_f[i][k]=2 
         
         if ((n_und_f[i][2]==1)&&(n_und_f[i][0]==1)):
            class_f[i][k]=2 
        
   else:
     for k in range(numberVar):
       for i in range(length_data -  sizeM):
         class_f[i][k] = 0 
       
 def MinVarFunc(length_data, f[length_data][ numberVar], *thres):
'''This function is used to replace an original time series with the sequence of ("1", "-1", "0") according to defined rules
   and given step-to-step variation threshold and finally with a class number assigned to every fixed time window  sizeM from a sequence

  thres_var - variation threshold
  f[length_data][ numberVar] - is an original time series
  length_data - is the length of time series,  numberVar is the number of variables used in the analysis (default  numberVar = 2)
  coarse_f[n- sizeM][ sizeM][ numberVar] - is resulting sequence of ("1", "-1", "0")
  class_f[length_data -  sizeM][ numberVar] - array of class numbers
'''

   # ep array measures number of positive "1", negative "-1" and no change "0" between
    neighbouring data points in time series 

   # finding maximum of time series
   ep = 0.0 
   for k in range(length_data):
       if (math.fabs(f[k][0]-f[k-1][0]) > ep):
          ep = math.fabs(f[k][0]-f[k-1][0]) 
       
   for i in range(numberVar):
     for k in range(length_data):
        #    printf("%g\n",fabs(f[k][i]-f[k-1][i])) 
       if ((math.fabs(f[k][i]-f[k-1][i]) < ep)&&(math.fabs(f[k][i]-f[k-1][i])>0)):
          ep = math.fabs(f[k][i]-f[k-1][i]) 
       #   printf("%g\t%g\t%g\n",ep,f[k][i],f[k-1][i]) 
       
   *thres=ep 

 def MatrixTransition(rows_m, columns_m, size_a, size_b, indx, **a, prob_trans[rows_m][columns_m]):
'''This function is used to calculate the transition probabilities prob_trans[rows_m][columns_m]
*  of a binary time sequence a[size_a][size_b] where rows_m = 2^ sizeM, columns_m = 2^ sizeM and
*   sizeM is a global constant defined as a fixed time window of the time sequence
*  columns of prob_trans matrix are normalized
'''
    for i in range(rows_m):
      for j in range(columns_m):
        prob_tr[i][j] = 0 
      

    for i in range(size_b  -  sizeM):
      for count in range(sizeM):
        seq_1[count] = a[indx][i + count] 
      
      BinaryInverse( sizeM, seq_1, &k1) 
      for count in range(sizeM):
        seq_2[count] = a[indx][i + count + 1] 
      
      BinaryInverse( sizeM, seq_2, &k2) 

      prob_tr[k2 - 1][k1 - 1] =  prob_tr[k2 - 1][k1 - 1] + 1 
    
    for j in range(columns_m):
      sum[j] = 0 
      for i in range(rows_m):
         sum[j] =  prob_tr[i][j] + sum[j] 
      
    for i in range(rows_m):
      for j in range(columns_m):
        if (sum[j]!=0):
         prob_tr[i][j] = prob_tr[i][j]/sum[j] 
        
     for i in range(rows_m):
      for j in range(columns_m):
         prob_trans[rows_m-i-1][columns_m-j-1] = prob_tr[i][j] 
      
 def PatternRecognition( no_opt, ind, decay_ker, epsil, \
                          length_data, t[ sizeBinF], f[length_data- sizeM][ sizeM][ numberVar], class_f[length_data- sizeM][ numberVar], total_r, \
                          total_c, *count_ind, **binary_code_f, **opt_seq_eps, **opt_seq_ker, \
                          **opt_seq_num, **w_seq):
''' This function performs binary coding of data sequence class_f (if classes option is chosen) or sequence f (if no classes                                                                                                               option
*  is selected) and implements pre-selection of binary codes according to defined weighting function WeightedSumOpt
*  The results of pre-selection procedure are saved in the array w_seq[ sizeBinF * rows_g][length_data -  sizeM]
*  as well as parameters opt_seq_eps[ sizeBinF * rows_g][0] , opt_seq_ker[ sizeBinF * rows_g][0] and
*  opt_seq_num[ sizeBinF * rows_g][0] corresponding to the threshold of variation, decay kernel and number of binary sequence correspondingly
*  The selection is done out of number of  sizeBinF*rows_f of possible binary coding rules
'''
   columns_f =  math.pow(3, sizeM) 
   rows_f = math.pow(2,columns_f) 
   columns_g = math.pow(2, numberVar) 
   rows_g = math.pow(2,columns_g) 
  
   opt_threshold = 1.0 
   *count_ind=0 

   value_dc = 20.0 

   char name_file_completebin = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/completebin6.dat" 
   char name_file_cbin = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/completebin.dat" 
 # open file for writing binary table F 

  
   file_completebin = open(name_file_completebin,"w") 
   file_cbin = open(name_file_cbin,"w") 



   BinaryMatrix(rows_g, columns_g, binary_code_g, binary_code_g_inv) 

   number_classes = 2 *  sizeM - 1 
   power_f = math.pow(2,number_classes) 

   
   BinaryMatrix(power_f, number_classes, bn_ sizeM, bn_inv) 

     count_i = 0 
     total_G_F =  sizeBinF * rows_g 
     total_size = math.pow(2, math.pow(3,  sizeM )) 


     for jj in range(rows_g):
       for i in range(sizeBinF):
# *********** Binary coding of an original sequence f ( flag_classes==0) or class_f ( flag_classes==1) *************** 

             if ( flag_classes == 0){
             for j in range(length_data -  sizeM):
                 for l in range(numberVar):
                  for ii in range(sizeM): 
                     s[ii]=f[j][ii][l] 
                  

                  FunctionF( sizeM, s, &f_inv) 

                  combined_data_t[l] = binary_code_f[i][f_inv-1] 
               
               BinaryInverse( numberVar,combined_data_t, &f_inv) 

               seq[j] = binary_code_g[jj][f_inv-1] 

        
            
            else:

             for j in range(length_data -  sizeM):
               for l in range(numberVar):

                  f_inv = class_f[j][l] 

                  combined_data_t[l] = bn_ sizeM[i][f_inv-1] 
                  file_completebin.write(combined_data_t[l]) 


               BinaryInverse( numberVar,combined_data_t, &f_inv) 

               seq[j] = binary_code_g[jj][f_inv-1] 
               file_cbin.write(seq[j]) 

            
              file_cbin.write(i +  sizeBinF*jj,i,jj + '\n') 
              file_completebin.write(i +  sizeBinF*jj,i,jj+ '\n') 
            

                if (*count_ind < total_G_F):
                   opt_seq_eps[*count_ind][0] = epsil 
                   opt_seq_ker[*count_ind][0] = decay_ker 
                   opt_seq_num[*count_ind][0] = ( flag_classes == 0) ? t[i] + total_size*jj : i +  sizeBinF*jj 
               
                   for j in range(length_data -  sizeM):
                     w_seq[*count_ind][j]=seq[j] 
                
                   *count_ind = *count_ind + 1 

       
  file_completebin.close() 
  file_cbin.close() 

 def BinaryInverse(size_f, f[size_f], *bin_to_int):
 ''' This function returns the row of a binary matrix
 * f[size_of(f)] by comparing it to the row of matrix F_bin[2^size_of(f)][size_of(f)]
 * bin_to_int is a number of row
 ''' 
   *bin_to_int = 0 
   for count_ind in range(size_f):
     *bin_to_int=f[size_f-1-count_ind]*math.pow(2,count_ind)+*bin_to_int 
   
   *bin_to_int = *bin_to_int + 1 
 
 def FunctionF( size_f, f[size_f], *bin_to_int):
 ''' This function returns the column of a binary matrix by comparing it to a given binary sequence
  f[size_of(f)] is a binary sequence that is compared to the column of matrix
 * F_bin[2^size_of(f)][size_of(f)]
 * bin_to_int is a number of row
 '''
   sum_F = 0 
   for i in range(size_f):
    sum_F = f[i] + sum_F 
   
   i = - size_f 
   
   while (i < size_f + 1):
     if (sum_F == i ):
       if (sum_F <= - size_f + 1 ):
         *bin_to_int  = 1
         
       else if ((sum_F > - size_f + 1 )&&(sum_F < size_f - 1)):
         *bin_to_int  = size_f + i  

       else if (sum_F >= size_f - 1 ):
         *bin_to_int  = 2*(size_f - 1) + 1 
       
     i ++ 
 
 def WeightedSumOpt( fi, funct[3], length_data, opt_threshold, decay_mk, res[length_data -  sizeM], *i_opt):
''' This function preselect the binary coding rules according to weighting function defined over the intervals of
* known recession periods
* The function returns *i_opt=1 if the input binary sequence res[length_data -  sizeM] satisfied the condition
'''
   *i_opt = 0 
   a = (length_recession + 1)/2 
   b = (length_recession + 1)/2 + length_pre_recession 
   total_length = length_pre_recession + length_recession + length_post_recession 
   length_2 = length_pre_recession + length_recession 
   length_1 = length_pre_recession  
   
   sum_1 = 0 
   denom_sum_1 = 0 
   sum_2 = 0 
   sum_3 = 0 
   denom_sum_2 = 0 
   denom_sum_3 = 0 

   sum0 = 0 
   for count_local in range(length_data- sizeM):
        sum0 = res[count_local]+sum0 
   
   for count_local in range(total_length):
     rules[count_local] = res[mid_point + count_local - 1] 
   
   for count_local in range(length_1 + 1): 
     w = exp((a - math.fabs(count_local - b))/decay_mk) 

     sum_1 = sum_1 + (1 - rules[count_local -1]) * (1 - w) 

     denom_sum_1 = denom_sum_1 + (1 - w) 
   
   for count_local in range(length_1+1,length_2+1):
     sum_2 = sum_2 + (1 - rules[count_local - 1]) 
   
   for count_local in range(length_2 +1,total_length + 1):
     w = math.fexp((a - math.fabs(count_local - b))/decay_mk) 
     sum_3 = sum_3 + (1 - rules[count_local - 1])*(1 - w) 

     denom_sum_3 = denom_sum_3 + (1 - w) 
   
   funct[0] = math.pow(sum_1/denom_sum_1,2) 
   funct[1] = math.pow(1-sum_2/length_recession,2) 
   funct[2] = math.pow(sum_3/denom_sum_3,2) 
  # printf("%g\t%g\t%g\n",funct[0],funct[1],funct[2]) 
# Assign the flag *i_opt=1 if the binary sequence satisfies the given conditions
#  if *i_opt==1 then the sequence is selected , otherwise it is rejected */
   if ((funct[0] < opt_threshold)&&(funct[1] < opt_threshold)&&(funct[2] < opt_threshold)):
     *i_opt = 1 

  
 def BinaryMatrix( rows_matr, columns_matr, **bn, **gn):
 '''Here a two-dimensional binary matrix bn[rows_matr][columns_matr] is initialized
 *  with rows_matr = 2^columns_matr
 * Example of a binary matrix of size 3 x 2^3
 *     0 0 0
 *     0 0 1
 *     0 1 0
 *     0 1 1
 *     1 0 0
 *     1 0 1
 *     1 1 0
 *     1 1 1
 '''
   k = 0
   for i in range(rows_matr):
     for j in range(columns_matr):
       bn[i][j] = 1 
     

   for i in [columns_matr:0:-1]:
     for j in [1 + pow(2,k):rows_matr + 1:pow(2,k+1)] :
       for l in range(pow(2,k)):
            bn[j + l - 1][i - 1] = 0 
       
     k = k + 1 
   
   for i in range(rows_matr):
             for j in range(columns_matr):
                    gn[rows_matr-1-i][j]=bn[i][j] 
    
 def BinaryMatrixR(rows_matr, columns_matr, t[ sizeBinF], **bn):
 '''  Here a two-dimensional binary matrix bn[ sizeBinF][columns_matr] is initialized
 *  with  sizeBinF <= 2^columns_matr
 * The matrix bn is constructed with rows from a full size binary matrix by selecting
 *  sizeBinF rows according to the sequence of numbers in t[ sizeBinF]
 '''
   k = 0, i, j, l = 0, ii = 0, k1 = 0 

     for j in range(columns_matr):
       for (k1 in range(sizeBinF):
          bn[k1][j] = 0 
       }
     }

   for i in [columns_matr:0:-1]:
     for j in [1 + pow(2,k):rows_matr + 1:pow(2,k+1)]:
       for l in range(pow(2,k)):
         for k1 in range(sizeBinF):
            if (t[k1] == j + l - 1):
              bn[k1][i - 1] = 1 
           
     k = k + 1 
   
 def BinaryMatrixF(rows_matr, columns_matr, k_row, k_col, *bn):
 '''  Here a two-dimensional binary matrix bn[rows_matr][columns_matr] is initialized
 *  with rows_matr = 2^columns_matr
 '''
   k = 0

   for i in [columns_matr:0:-1]:
     for j in [1 + math.pow(2,k):rows_matr + 1:math.pow(2,k+1)]:
       for l in range(math.pow(2,k)):
            if ((k_row == j + l - 1)&&(k_col == i - 1)):
              *bn = 1 
            
            else:
              *bn = 0 
           
     k = k + 1 
   
 def Contains(arr[], n, x):
# the function returns 1 if x is in array arr[n], otherwise 0
#
    while (n--):
        if (arr[n] == x):
           return 1 
        n = n - 1

    return 0 

 def AssignTransProbabilities(rows, cols, row_trans, **b, **matr_T, **trans_matr_T, \
                      prob[row_trans][row_trans]):
''' This function makes use of probability of transitions matrix prob[row_trans][row_trans] and fills the
* matrix of transition probabilities trans_matr_T[rows][cols]
'''
   for ind_2 in range(rows):
   for ind_1 in range(cols):
       for i in range(sizeM):
         seq_1[i] = matr_T[ind_2][ind_1 + i] 
         seq_2[i] = matr_T[ind_2][ind_1 + 1 + i] 
       
       BinaryInverse( sizeM, seq_1, &k1) 
       BinaryInverse( sizeM, seq_2, &k2) 
       trans_matr_T[rows-ind_2-1][ind_1] = prob[k2 - 1][k1 - 1] 
    
 def SummedProbability(nums, row_trans, q_0, pred_amb, time_length, p_threshold, count_seq, length_seq, **seq, \
                       prob_matr[row_trans][row_trans], **trans_matr_T, **b_matrix, **matr_T, w_flag, *f_j, *fnopostrec_j,\
                        *fnopost_j):
''' Integrate probabilities over intervals corresponding to recession, pre-recession and post-recession
* using given binary sequence seq[][]
* The integral value or summed probabilities are written into output *f_j
'''
   rows_T = math.pow(2,time_length) 
   
   test = 0 
  
   for i in range(rows_T):
     r = 1 
     for j in range(time_length):
       r = r*math.pow(1-q_0,matr_T[i][j]) 
   #      r = r*pow(1-q_0*pow(p_threshold,time_length-j),matr_T[i][j]) 
     
     p_T[i]=r 
   

   sum_1=0.0 
   sum_2=0.0 
   sum_3=0.0 

   interval_pre_post = length_post_recession + length_pre_recession 
   
   for i in range(length_recession + 1):
     for j in range(pred_amb + 1):
       ind1 = start_recession + i - 1 - j 
       ind2 = start_recession + i - 1 
       ProductTransProbability(test, row_trans, ind1, ind2, time_length, rows_T, count_seq, length_seq, seq, prob_matr,matr_T,p_T,trans_matr_T,b_matrix,&prob) 
       if (prob <= p_threshold):
         sum_1 = sum_1 + 1 
    
   for i in range(length_pre_recession + 1):
     for j in range(pred_amb + 1):
       ind1 = start_recession - i - j 
       ind2 = start_recession - i 
       ProductTransProbability(test, row_trans, ind1, ind2, time_length, rows_T, count_seq, length_seq, seq, prob_matr,matr_T,p_T,trans_matr_T,b_matrix,&prob) 
       if (prob > p_threshold):
         sum_2 = sum_2 + 1 
       
   for i in range(length_post_recession + 1):
     for j in range(pred_amb + 1)
       ind1 = start_recession + length_recession + i - 1 - j 
       ind2 = start_recession + length_recession + i - 1 
       ProductTransProbability(test, row_trans, ind1, ind2, time_length, rows_T, count_seq, length_seq, seq,prob_matr,matr_T,p_T,trans_matr_T,b_matrix,&prob) 
       if (prob > p_threshold):
         sum_3 = sum_3 + 1 
       
   *f_j=(sum_1 + sum_2 + sum_3)/(pred_amb*total_length) 
   *fnopost_j=(sum_1 + sum_2)/(pred_amb*(total_length-length_post_recession)) 
   *fnopostrec_j=(sum_2)/(pred_amb*length_pre_recession) 

 def ProductTransProbability(test, row_trans, ind_1, ind_2, cols_matrix_T, rows_matrix_T, count_seq, \
                              length_seq, **seq, prob_matr[row_trans][row_trans], **matr_T, *p_T, **trans_matr_T, **b_mat, *product_prob):
''' The function estimates the entries of probability of recession *product_prob based on distance ind_2-ind_1
 *  between current month ind_1 and predicted month ind_2  the probability is evaluated based on matrix prob_matr and input
 * binary sequence seq[ sizeBinF*rows_g][length_data-  sizeM], here count_seq <  sizeBinF*rows_g is a number of binary sequence
 * selected out of total  sizeBinF*rows_g sequences
 ''' 
  inter_length = ind_2 - ind_1 


  *product_prob = 0 

# Probability estimated for matrix entries with prediction distance (or difference between current month and predicted month)
# greater or equal to T - 1*/
  if (inter_length >= cols_matrix_T - 1):

    for count in range(sizeM):
      s_1[count] = seq[count_seq][ind_1 -  sizeM + count] 
    
    BinaryInverse( sizeM, s_1, &r) 
    r=math.pow(2, sizeM)-r+1 

    for count in range(row_trans):
      pr[count] = 0 
    

    pr[r-1] = 1 
    PowMatrix(row_trans, prob_matr, prob_out, inter_length - cols_matrix_T +2) 

    for count_rows in range(row_trans):
           pp[count_rows] = 0 
        for count_cols in range(row_trans):
           pp[count_rows] = pp[count_rows] + prob_out[count_rows][count_cols]*pr[count_cols] 
        

    
    for i in range(row_trans):
        sum[i] = 0 
    

    for ie in range(rows_matrix_T):
      prod = 1 
      for je in range(cols_matrix_T -  sizeM):
        prod = prod * trans_matr_T[ie][je] 
      
      prod = prod * p_T[ie] 

      for ke in range(row_trans):
        flag = 0 
          for i in range(sizeM):
            if (matr_T[ie][i]==b_mat[ke][i]):
              flag = flag + 1 
            
        if (flag ==  sizeM):

                sum[ke] = sum[ke] + prod 

        
    p_c = 0 
    for ie in range(row_trans):
      p_c = p_c + pp[ie]*sum[ie] 


 # Probability estimated for matrix entries with prediction distance (or difference between current month and predicted month)
 # not exceeding T - 2
  else if ((inter_length >= 1)&&(inter_length <= cols_matrix_T - 2 )):

    for i in range(cols_matrix_T  + 1 - inter_length):
      s_2[i] = seq[count_seq][i + ind_2 - cols_matrix_T - 1] 
    
    p_c = 0 

    for i in range(rows_matrix_T):
      flag = 0 
      for ie in range(cols_matrix_T - inter_length + 1):

        if (s_2[ie] == matr_T[i][ie]):
          flag= flag + 1 
        
      if (flag == cols_matrix_T - inter_length + 1):
            z = 1 
            if (inter_length >= 2):

              for j in range(inter_length - 1): 
                z = z * trans_matr_T[i][j + cols_matrix_T - 2 - inter_length] 
              
            z = z * p_T[i] 
            p_c = p_c + z 
      
  *product_prob = p_c 
 
 def OptimalParameters(*opt_thres, *opt_q, *opt_J, *opt_p, *opt_ind, length_amb, length_data, time_length, row_trans, count_seq, q[size_q], opt_seq[length_data -  sizeM], **seq, \
                         **opt_seq_eps, **opt_seq_ker, **opt_seq_num, **b_matrix, prob_tr[row_trans][row_trans], w_false):
''' This function is used to run optimization procedure and to estimate the optimal parameters *opt_thres, *opt_q, *opt_p, *opt_ind
* as well as value of subjective function *opt_J from a given sequence seq[ sizeBinF*rows_g][length_data- sizeM]
* The optimal binary sequence that fits the optimization criteria *opt_seq[length_data- sizeM] is found
'''
   number_rows = pow(2,time_length) 
   number_cols = time_length -  sizeM 


   BinaryMatrix(number_rows, time_length, matrix_T, binary_code_inv) 

   for count_s in range(count_seq):

   if (TRANS_TOTAL == 0):
   #  if TRANS_TOTAL == 0 option is chosen than matrix of transitions should be calculated for every sequence seq
     MatrixTransition(row_trans, row_trans, count_seq, length_data -  sizeM, count_s, seq, prob_trans) 
   
   else:
   # if TRANS_TOTAL == 1 option is chosen than the matrix of transition is given based on entire time series
     probability(row_trans, row_trans, prob_trans, prob_total) 
   
   for i in range(row_trans):
    for j in range(row_trans):
        pr_tr[i][j][count_s]=prob_trans[i][j] 
 

     AssignTransProbabilities(number_rows, number_cols, row_trans, b_matrix, \
                                   matrix_T, transition_matrix, prob_trans) 

     q_0 = 0.0, p_threshold = 0.0 
     #   p_threshold = 0  # always no-recession is assumed
     #  p_threshold = 1  # always recession is assumed
# **************Start of search for minimum of subjective function ******************* 
     SummedProbability(opt_seq_num[count_s][0],row_trans, q_0, length_amb, time_length, \
                    p_threshold, count_s, length_data- sizeM, seq, prob_trans, transition_matrix, b_matrix, matrix_T, w_false, &f_j, &fpostrec_j, &fpost_j) 

     J_opt = f_j 
     Jpost_opt = fpost_j 
     Jpostrec_opt = fpostrec_j 
     q_opt=0.0, p_threshold_opt=0.0 

     for count_1 in range(size_q):
        q_0 = q[count_1] 
       for count_2 in range(size_q):
         p_threshold = q[count_2] 
       #   p_threshold = 0  // always no-recession is assumed
       #  p_threshold = 1  // always recession is assumed

         SummedProbability(opt_seq_num[count_s][0],row_trans, q_0, length_amb, time_length, \
                    p_threshold, count_s, length_data- sizeM, seq, prob_trans, transition_matrix, b_matrix, matrix_T, w_false, &f_j, &fpostrec_j, &fpost_j) 

         J = f_j 
         Jpost = fpost_j 
         Jpostrec = fpostrec_j 


         if (J < J_opt):
           J_opt = J 
           q_opt = q_0 
           p_threshold_opt = p_threshold 
       

     result[count_s][0] = J_opt 

     result[count_s][1] = q_opt 
     result[count_s][2] = p_threshold_opt  
   

   MinimumFromArray(count_seq, 3, result, &J_0, &min_index) 


   *opt_ind = opt_seq_num[min_index][0] 
   *opt_thres = opt_seq_eps[min_index][0] 
   *opt_q = result[min_index][1] 
   *opt_p = result[min_index][2] 
   *opt_J = J_0 

   opt_ker=opt_seq_ker[min_index][0] 
  
   char name_file_bin = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/binfile.dat" 
 # open file for writing binary table F */

   file_bin_seq = open(name_file_bin,"w") 
   
   for k in range(length_data - sizeM-1):
    opt_seq[k] = seq[min_index][k] 
    file_bin_seq.write(opt_seq[k]) 
   
   opt_seq[length_data -  sizeM-1] = seq[min_index][length_data -  sizeM-1] 
    file_bin_seq.write(opt_seq[length_data -  sizeM-1]) 
    file_bin_seq.close() 
#   End of optimization procedure, minimum of objective function *opt_J is found ******************* 
  for k in range(count_seq):
  if (k==min_index):
  for i in range(row_trans):
    for j in range(row_trans):
        prob_tr[i][j]=pr_tr[i][j][k] 
   
 def MinimumFromArray( rows_f, columns_f, f[rows_f][columns_f], *min_value, *min_index):
'''This function finds a minimum element in a given column of two--dimensional matrix and returns index of
  of the element and matrix value
  f[rows_f][columns_f] is matrix and its minimum value in the first column is *min_value
  *min_index is the index of the element
'''
   *min_value = f[0][0] 
   *min_index = 0 
   for i in range(rows_f):
     if (f[i][0] < *min_value):
       *min_value = f[i][0] 
       *min_index = i 
 
 def MinimumF(size_f, f[size_f], *min_index):
'''This function finds a minimum element in a one--dimensional array and returns index of
  of the element
  f[size_f] is matrix and its minimum value is f[*min_index]
  *min_index is the index of the element
'''
  *min_index = 0 
  f_min=f[0] 
  for i in range(size_f):
    if (f_min > f[i]):
      f_min = f[i] 
      *min_index = i 
  
 def PowMatrix(n, matrix_in[n][n], matrix_out[n][n], d):
'''This function defines power of two-dimensional square matrix  and return the resulting two dimensional matrix
  matrix_in[n][n] is input matrix of size n x n
*  matrix matrix_out[n][n] is an output matrix ,
*  d is the power
'''
    for i in range(n):
      for j in range(n):
         temp_in[i][j] = matrix_in[i][j] 
     
    for ss in range(d - 1):
    for i in range(n):
      for k in range(n):
        temp_out[i][k] = 0 
        for j in range(n):
            temp_out[i][k] = temp_out[i][k] + temp_in[i][j]*matrix_in[j][k] 
     
    for i in range(n):
      for j in range(n):
        temp_in[i][j] = temp_out[i][j] 
     
    for i in range(n):
      for j in range(n):
         matrix_out[i][j] = temp_in[i][j] 

 def DefineMatrixProbabilities(row_trans, time_length, q_0, p, count_s, length_seq, **seq_t, \
      prob_matr[row_trans][row_trans], **matr_T, **trans_matr_T, **b_mat, opt_p, **matrix_p:
''' This functions evaluates matrix of recession matrix_p[interval_current+2][interval_predicted+2]
* using initial given binary sequence seq_t[ sizeBinF*rows_g][length_seq]
'''
   
   rows_T = math.pow(2,time_length) 
   
   for i in range(rows_T):
     r = 1 
     for j in range(time_length):
       r = r * math.pow(1-q_0*math.pow(p,time_length - j),matr_T[i][j]) 
     
     p_T[i] = r 
   

   for i in range(interval_current + 2):
      for j in range(interval_predicted + 2): 
        matrix_p[i][j] =float('nan') 
    
   test =1  
   for i in range(interval_predicted + 1):
     for j in range(interval_current + 1): 
       if (( start_current_period + j) < ( end_predicted_period - i)) {
         ProductTransProbability(test, row_trans, start_current_period + j,end_predicted_period - i , \
                                 time_length, rows_T, count_s, length_seq, seq_t, prob_matr, matr_T, p_T, trans_matr_T, b_mat, &product_prob) 
            matrix_p[interval_current - j][interval_predicted - i] = product_prob 

 
 def probability(rows, cols, prob[rows][cols], prob_total[rows][cols]):
 ''' This function assigns transition probability matrix from a given prob_total if option TRANS_TOTAL 1
   is chosen (probability is evaluated using entire time series)
 '''
  for i in range(rows):
    for j in range(cols):
       prob[i][j]=prob_total[i][j] 

# ***********  Open files for reading recession data and writing recession parameters ******************************************

   start = clock() 


   row_trans =math.pow(2, sizeM) 

   char c1[3] 

   strcpy(c1, "") 
   sprintf(c1, "%s", argv[1]) 


   name_file_data = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/Rec6/DataRec" 
 # open file for reading recession data 

   
   strncat(name_file_data =name_file_data + c1 + ".dat" 


   file_data_recession = open(name_file_data,"r+") 
   


   name_file_opt_parameter = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/optimal_parameters_" 
   # open file for writing an optimal parameters defined from optimization procedure 

   name_file_opt_parameter = name_file_opt_parameter + c + "_" + cl + ".dat"


   file_optimal_parameter = open(name_file_opt_parameter,"w") 

   f_step = 0 
   ch = 0 
   
   file_data_recession.read(st) 
   
   file_data_recession.read(&length_pre_recession, &length_recession, &length_post_recession, &mid_point) 
   total_length = length_pre_recession + length_recession + length_post_recession 
   start_recession = mid_point + length_pre_recession 

   #while ((ch = fgetc(file_data_recession)) != EOF){
     count_col = 0 
     for count_col in range(numberVar-1)):
       file_data_recession.read( &data_1) 
       printf("%g\t", data_1) 
    
       file_data_recession.read(&data_1) 
       
       f_step++
       
   length_data = f_step 

  # rewind(file_data_recession) 
   file_data_recession.read(st) 
  # printf("%s\n", st) 

   f_step = 0 
   file_data_recession.read(&length_pre_recession, &length_recession, &length_post_recession, &mid_point) 
   while f_step < length_data:

     count_col = 0 
     for count_col in range(numberVar):
      file_data_recession.read(&data_1) 
         data_recession[f_step][count_col] = -data_1

     file_data_recession.read(&data_1) 
     data_recession[f_step][ numberVar-1] = -data_1 
     f_step = f_step + 1 

# *********************End of reading recession data ****************************************************************** 

   total_rows = math.pow(2,math.pow(2, numberVar)) 
   intotal_columns = math.pow(2,math.pow(3, sizeM)) 

    sizeBinF = ( flag_classes == 1) ? math.pow(2,2* sizeM - 1) : MAX_INT  # if classes option is chosen   flag_classes ==1 the binary table F has number of rows 2^(2* sizeM-1)
                                                                       #   otherwise the number of columns equal to MAX_INT 

   rows_g = math.pow(2,math.pow(2, numberVar)) 
   total_size_f_g = rows_g *  sizeBinF 
   
   num_total_opt =  sizeBinF * size_eps * size_dk *rows_g 


   num_seq = 0 
   count_opt = 0
   value_eps = 0.04 

   columns_f =  math.pow(3, sizeM) 
   rows_f = math.pow(2,columns_f) 

   ind_select = 1 
   ind=0 
   no_opt = 0 

   count_seq = 0 

   char name_file_F = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/matrix_F_" 
 # open file for writing binary table F 

   strcat(name_file_F = name_file_F + c + ".dat" 
  
   file_matrix_trans = open(name_file_F,"w") 

   char name_file_coarsegrain = "C:/Users/shchekin/Documents/ElenaDocuments/InclinationAnalysis/InclinationAlgorithmTotal/DataElisabeth/coarsegrain.dat" 
 # open file for writing binary table F */

   file_coarsegrain = open(name_file_coarsegrain,"w") 

   count_ind = 0 
   j = 0 
   
# ***********************  Start of pre--selection of parameters and binary sequence that satisfy inner optimization criterion ********    
   for count_eps in range(size_eps): # iterate over index representing length of threshold eps 

     GridNew(length_data,data_recession, coarse_data, eps[count_eps],class_f, &var_thres)  # coarse--graining of data according to the threshold eps
# data_recession is initial time series, coarse_data is a sequence of {-1, 0, 1} formed from initial time series according to the threshold eps 

     for i in range(numberVar):
        for j1 in range(length_data- sizeM-1):
          file_coarsegrain.write(coarse_data[j1][0][i]) 
    
        file_coarsegrain.write(coarse_data[length_data- sizeM-1][0][i],var_thres) 
     
     epsil = var_thres 
 

     for count_ker in range(size_dk): # iterate over index representing length of decay_kernel 

       decay_ker = decay_kernel[count_ker] 

       PatternRecognition(no_opt, ind, decay_ker, epsil, length_data, t, coarse_data, class_f,\
                             total_rows, total_columns,  &count_ind, binary_code_f, seq_eps, seq_ker, seq_num, w_seq) 

       for i in range(count_ind):
         opt_seq_eps[i + j][0]=seq_eps[i][0] 
         opt_seq_ker[i + j][0]=seq_ker[i][0] 
         opt_seq_num[i + j][0]=seq_num[i][0] 
         for jj in range(length_data -  sizeM):
           opt_w_seq[i + j][jj]=w_seq[i][jj] 
         
       j = j + count_ind 
    
   count_ind = j 
   file_coarsegrain.close() 
# ******************  End of pre--selection of parameters and binary sequence that satisfy inner optimization criterion ***************   

     BinaryMatrix(row_trans,  sizeM, b_matrix, g_matrix)  # define a binary matrix with the number of columns  sizeM

     file_optimal_parameter.write("var_f", "eps_f","q_f", "J_f", "f_f", "p_f", "T" ,"L") 

     count_ind = j 
 # ***** Start of optimization procedure during training stage *************************************************************************/
     for count in range(sizeT): # iterate over index representing length of prediction base parameter

       time_length = T[count] 

       for l in range(size_amb): # iterate over index representing length of prediction ambition 

         length_amb = amb[l] 

         OptimalParameters(&opt_thres, &opt_q, &opt_J, &opt_p, &opt_f, length_amb, length_data, time_length, row_trans, count_ind, q, opt_seq, opt_w_seq,\
                             opt_seq_eps, opt_seq_ker, opt_seq_num, b_matrix, prob_trans,1) 
         thres_f[count][l]= opt_thres 
         q_f[count][l]= opt_q 
         J_f[count][l]= opt_J 
         i_f[count][l]= opt_f 
         p_f[count][l]= opt_p 
         Jmin=J_f[0][0] 

# ***************** writing to output file and print of optimization parameters and objective function J_f ************************************** */

         file_optimal_parameter.write(var_thres, thres_f[count][l],q_f[count][l], J_f[count][l], i_f[count][l], p_f[count][l], time_length, length_amb) 

# ****** writing to output file and print of the binary sequence that was identified in optimization ********************************************** */

         file_matrix_trans.write(J_f[count][l]) 
         
'''* **************** End of optimization procedure ********************************************************* */
////
///* ****************** Close data files ********************************************************************** '''
     file_data_recession.close() 
     file_optimal_parameter.close() 
     file_matrix_trans.close() 


