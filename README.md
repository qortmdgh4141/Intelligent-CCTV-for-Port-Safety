# 📹  Intelligent CCTV for Port Safety 
<br/> 
 
### 1. &nbsp; Background of Development <br/><br/> 

- _Recently, CCTV-related technologies have been used in various ways in our daily lives, such as security and safety accident prevention. However, traditional CCTV cameras record more unnecessary information than is needed when a problem occurs. In addition, it is difficult to fully recognize and judge the site only with existing CCTV cameras. The Korea Safety and Health Agency announced that six deaths occurred every year at domestic ports during 2011-2021. It shows the limitations that existing CCTV cameras cannot solve safety accidents and human casualties in domestic ports. In order to solve these limitations, I devised ""Intelligent CCTV for Port Safety & Real-Time Information Provision System" that can quickly and accurately respond to dangerous situations while checking the site in real time._ <br/><br/><br/>

### 2. &nbsp; Project Introduction <br/><br/>   

- _The project used a variety of object detection and action recognition models based on traditional computer vision technique and deep neural network._ <br/><br/>

- _Based on the various deep learning models mentioned above, the following algorithms are implemented :_ <br/>
  
  - _Region of Interest (ROI)_ <br/>
    - _Regions of interest (ROI) means a meaningful and important regions in an images._ <br/>
    - _The ROI algorithm is implemented based on a binary mask, one of image processing techniques._ <br/>
    - _In the mask image, pixels that belong to the ROI are set to 1 and pixels outside the ROI are set to 0._ <br/> 
    
  - _DeepSort_ <br/> 
    - _DeepSort is the tracking algorithm which tracks objects not only based on the velocity and motion of the object but also the appearance of the object._ <br/>
    
  - _Measuring Distance Between Objects_ <br/>
    - _The algorithm detects and analyzes low-risk and high-risk state by measuring the distance between the bounding box centers of objects._ <br/>
    - _When measuring distance, measure the distance between all objects detected in the image by reflecting the characteristics of a complete graph._ <br/>

  - _Time Series Data Analysis_ <br/>
    - _This algorithm detects and analyzes time series data using a queue, and its contents are as follows :_ <br/>
    
      1. _If the data recognized through the action recognition model is judged to be abnormal behavior, the penalty score is sequentially stored in the queue, which is a linear data structure. (Conversely, if the recognized data is judged to be normal behavior, the advantage score  is stored.)_ <br/>
      2. _At the same time, scores, which are time series data previously stored in the queue, are deleted from the queue by the FIFO (First In First Out) method of the queue._ <br/>    
      3. _By analyzing the sum of the scores in the queue in real time, if the value is an outlier, it is judged that it is currently a very dangerous situation_.<br/><br/>
  
- _Through the algorithms mentioned above, the following events are detected and analyzed :_ <br/>

  - _Intrusion and Loitering_ <br/> 
  - _Risk of Access to Restricted Areas_ <br/> 
  - _Risk of Collision Between Workers_ <br/> 
  - _Risk of Not Wearing Worker Safety Equipment_ <br/> 
  - _Fire Accident_ <br/> 
  - _Act of Smoking_ <br/> 
  - _Act of Falling_ <br/> 
  - _Act of Violence_ <br/><br/>

- _The analyzed information is stored in the database, and the administrator can quickly identify the current situation through text and graph-type information provided in real time._ <br/>

  - _The information can be checked not only on PC but also on smartphone._ <br/><br/>

- _If these functions are applied to the integrated control center in the port, a smart port monitoring system capable of efficient management and operation can be established._ <br/><br/><br/>
  
### 3. &nbsp; Main Function <br/><br/>

- _**Region of Interest (ROI)**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/region_of_interest_(roi).png?raw=true"  width="1280" height="340"> <br/>

  - _The administrator can set the ROI to a rectangle or polygon shape by dragging or clicking the mouse._ <br/>  
  
  - _Object detection is processed only within the Red ROI Border._ <br/> 
    - _Green ROI Border &nbsp;&nbsp; : &nbsp; Specify_ <br/>
    - _Yellow ROI Border &nbsp; : &nbsp; Modify_ <br/>
    - _Red ROI Border &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : &nbsp; Setup Complete_ <br/>
  
  - _Through this, the administrator can improve the processing speed and accuracy of object detection by removing unnecessary areas from the image._ <br/><br/><br/>
        
 - _**Intrusion and Loitering**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/intrusion_loitering.png?raw=true"  width="1280" height="230"> <br/>  

    - _The object detection model detects human intrusion, and if the intruder stays for a long time, it is judged that intruder is  loitering._ <br/>
      - _Intruder &nbsp; : &nbsp; The 'Intrusion' text is displayed in green at the top of the bounding box._ <br/>
      - _Loiterer &nbsp; : &nbsp; The 'Pedestrian loitering' text is displayed in purple at the top of the bounding box._ <br/>

    - _Even if the intruder and loiterer appear again after being covered by another object or going out of the video, they are recognized as the same person because the DeepSort algorithm has been applied._ <br/>
      - _First, when an intruder appears, the intruder is given a unique ID number, and the intruder's information is stored in the database with the ID number._ <br/>
      - _When the intruder reappears, it is recognized as the same person by the DeepSort algorithm and given the unique ID number previously given._ <br/>
      - _It then applies the previous information about the intruder by querying the unique ID number from the database._ <br/>
      
   - _Through this, the administrator can individually detect and analyze whether many people in the port are intruding and loitering._ <br/><br/><br/>

 - _**Risk of Access to Restricted Areas**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_access_to_restricted_areas.png?raw=true"  width="1280" height="220"> <br/> 

    - _The administrator can set the restricted area to a rectangle shape by dragging the mouse._ <br/>
        - _When the restricted area setting is completed, the restricted area is displayed as a white bounding box._ <br/> 
        
    - _Based on the algorithm of Measuring Distance Between Objects, when a person approaches the restricted area, a warning or danger message is sent to the administrator._ <br/>
        - Low-Risk State &nbsp;&nbsp; : &nbsp; The border of the bounding box is displayed in yellow.  <br/>        
        - High-Risk State &nbsp; : &nbsp; The border of the bounding box is displayed in red. <br/>
        
    - _Through this, the administrator can proactively block people from entering restricted areas within the port._ <br/><br/><br/>
        
 - _**Risk of Collision Between Workers**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_collision_between_worker.png?raw=true"  width="1280" height="160"> <br/>

    - _Based on the algorithm of Measuring Distance Between Objects, if there is a possibility that the safe distance between workers is not secured, a warning or danger message is sent to the administrator._ <br/>
        - _Low-Risk State &nbsp;&nbsp; : &nbsp; The safe distance between workers is displayed in yellow._ <br/>
        - _High-Risk State &nbsp; : &nbsp; The safe distance between workers is displayed in red._  <br/>
     
    - _Through this, the administrator can prevent collision accidents caused by the failure of workers to secure the safe distance in a dense space._ <br/><br/><br/>
        
 - _**Risk of Not Wearing Worker Safety Equipment**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_not_wearing_worker_safety_equipment.png?raw=true"  width="1280" height="440"> <br/>

    - _First, the object detection model detects a worker, a safety helmet, and a safety vest._ <br/>
    
    - _Next, based on the algorithm of Measuring Distance Between Objects, it analyzes whether the worker wears safety equipment._ <br/>
    
    - _If the worker is not wearing safety equipment, a warning or danger message is sent to the administrator._ <br/>
         - Low-Risk State &nbsp;&nbsp; : &nbsp; The Δ symbol is displayed in yellow in the bounding box.<br/>
         - High-Risk State &nbsp; : &nbsp; The X symbol is displayed in red in the bounding box. <br/>
         
    - _Through this, the administrator can prevent safety accidents caused by workers not wearing safety equipment at the work site._ <br/><br/><br/>

 - _**Act of Smoking**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/act_of_smoking.png?raw=true" width="1280" height="230"> <br/> 
   
    - If the action recognition model recognizes the behavior of smoking or lighting a cigarette, a danger message is sent to the administrator. <br/> 
        -  The 'Smoking Action' text is displayed in red at the top of the bounding box. <br/>
        -  The bounding box filled in purple is displayed. <br/>
          
    - _Through this, administrator can prevent fire accidents by quickly stopping people who smoke or light cigarettes in the hazardous areas within the port._<br/><br/><br/>

 - _**Act of Falling**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/act_of_falling.png?raw=true"  width="1280" height="160"> <br/> 

    - _First, the action recognition model recognizes the behavior of a worker who have fallen due to a fall accident in real time._ <br/>

    - _Next, based on the algorithm of Time Series Data Analysis, the current situation is judged as a safety stage, a warning stage, or a danger stage._ <br/>

      - _The warning stage is a situation where it is judged that a minor accident has occurred._ <br/>
          1.  The 'Warning Action' text is displayed in orange at the top of the bounding box. <br/>
          2.  The bounding box filled in orange is displayed. <br/>

      - _If it is judged that the injury is so serious that the worker cannot move, the warning stage is  converted to the danger stage._  <br/>
          1.  The 'Dangerous Action' text is displayed in red at the top of the bounding box. <br/>
          2.  The bounding box filled in red is displayed. <br/>

      - _After converting to the warning or danger stage, if the worker has regained consciousness and stands up, it is converted back to the safety stage._ <br/>    

    - _Through this, administrator can prioritize and respond to more dangerous situations even if multiple accidents occur simultaneously._ <br/><br/><br/>

 - _**Act of Violence**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/act_of_violence.png?raw=true"  width="1280" height="160"> <br/> 

    - _First, the action recognition model recognizes the behavior of making violent contact with another person's body in real time._ <br/>

    - _Next, based on the algorithm of Time Series Data Analysis, the current situation is judged as a safety stage, a warning stage, or a danger stage._ <br/>

      - _The warning stage is a situation that is judged to be minor violence or contact._ <br/>
            
          1.  The 'Warning Action' text is displayed in orange at the top of the bounding box. <br/>
          2.  The bounding box filled in orange is displayed. <br/>

      - _If it is judged that the violence is serious and needs to be restrained, the warning stage is converted to a danger stage._  <br/>
          1.  The 'Dangerous Action' text is displayed in red at the top of the bounding box. <br/>
          2.  The bounding box filled in red is displayed. <br/>

      - _After converting to the warning or danger stage, if violent behavior is restrained and not recognized, it is converted back to the safety stage._ <br/>    

    - _Through this, administrator can identify and restrain violent situations early._ <br/><br/><br/>

### 4. &nbsp; Real-Time Information Provision System <br/><br/>
  <img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/real_time_graph.png?raw=true"  width="640"> <br/>
  
  - _Since safety accidents occur at unexpected moments, it is important to check the site in real time and take prompt action._ <br/> <br/>
    
  - _Therefore, this project provides an information provision system that allows managers to check the site in real time._ <br/> <br/>
  
  - _First, after analyzing image data collected through Intelligent CCTV for Port Safety the following information is stored in a database._ <br/> 
  
    - Image of the On-Site <br/> 
    - The Number of People at the On-Site <br/>
    - Safety Numerical Values at the On-site <br/>
    - Identification Number of People at the On-Site <br/>
    - Type of Event <br/>
    - Occurrence Time of Event <br/>
    - Warning and Danger Stage of Event <br/><br/>
  
  - _Then, the information stored in the database is provided to the administrator's PC monitor or application in the form of text, image, graph, etc._ <br/>
    - _You can obtain the source code and information for the application from the following repository: <br/> https://github.com/qortmdgh4141/Real-time_Information_Provision_App.git_ <br/>
    
  - _Through this, administrator can check the situation in real time anywhere, not limited to places, and respond quickly to problems in the site._ <br/> <br/> <br/>
  
### 5. &nbsp; YOLO Model Training Strategies Using Transfer-Learning & Fine-Tuning <br/><br/>

- _**Transfer-Learning & Fine-Tuning Definition**_ <br/>

  - _Transfer learning consists of taking features learned on one problem, and leveraging them on a new, similar problem._ <br/>
  - _Transfer learning is usually done for tasks where your dataset has too little data to train a full-scale model from scratch._ <br/>
  - _The most common incarnation of transfer learning in the context of deep learning is the following workflow._ <br/>
    1. _Take layers from a previously trained model._ <br/>
    2. _Freeze them, so as to avoid destroying any of the information they contain during future training rounds._ <br/>
    3. _Add some new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset._ <br/>
    4. _Train the new layers on your dataset._ <br/>
  - _A last, optional step, is fine-tuning, which consists of unfreezing the entire model you obtained above (or part of it), and re-training it on the new data with a very low learning rate._ <br/><br/><br/> 
  
 - _**My Fine-Tuning Strategy**_ <br/> <br/>
 <img src="https://github.com/qortmdgh4141/AI_Lost_Pet_Search_App/blob/main/image/transfer_learning_fine_tuning_2.png?raw=true"  width="1320" height="490"> <br/>
   
|Strategy|Method |Feature|
|:-----------------------:|:--------------:|:-----------------------:|
|_**Strategy 1**_|_Train the entire model. In this situation, it is possible to use the architecture of the pre-trained model and train it according to the dataset._|_It is recommended for large datasets._|
|_**Strategy 2**_|_Train some layers and leave the others frozen. In a CNN architecture, lower layers refer to general features (problem independent), while higher layers refer to specific features (problem dependent). In this case, we have to adjust the weights of the network._|_This option is useful when we have a small dataset and a large number of parameters, we need to leave more layers frozen to avoid overfitting. On the other hand, if the dataset is large and the number of parameters is small, it is possible to improve the model by training more layers to the new task._|
|_**Strategy 3**_|_Freeze the convolutional base. In this situation, we have an extreme case of the train/freeze trade-off. The rationale behind it is to keep the original form of the convolutional base to use as input for the classifier. By this way, the pre-trained model plays the role of a feature extractor._|_It can be interesting for small datasets or if the problem solved by the pre-trained model is similar to the one we are working on._| 

<br/><br/> 


 - _**Results Based on 3 Strategies**_ <br/> <br/>
  <img src="https://github.com/qortmdgh4141/AI_Lost_Pet_Search_App/blob/main/image/transfer_learning_fine_tuning.png?raw=true"  width="1280" height="340"> <br/> <br/> 
    - _**Strategy 1** &nbsp; : &nbsp; Yellow, &nbsp;&nbsp;&nbsp;&nbsp; **Strategy 2 &nbsp; :** &nbsp; Pink, &nbsp;&nbsp;&nbsp;&nbsp; **Strategy 3 &nbsp; :** &nbsp; Purple_  <br/> 
    - _Strategy 1 shows the best results._ <br/>
    - _I think the reasons for this result are as follows._ <br/>
    - _The dataset I used is a large dataset and has little resemblance to the dataset of pre-trained models_ <br/> <br/> <br/>
    
--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Visual Studio-5C2D91?style=flat-square&logo=Visual studio&logoColor=white"/> 
  <img src="https://img.shields.io/badge/CMake-A0A0A0?style=flat-square&logo=CMake&logoColor=064F8C"/>
</p>  
<p>
  <img src="https://img.shields.io/badge/PyCharm-66FF00?style=flat-square&logo=PyCharm&logoColor=black"/>
  <img src="https://img.shields.io/badge/NVIDIA-black?style=flat-square&logo=NVIDIA&logoColor=76B900"/>
  <img src="https://img.shields.io/badge/MySQL-00CCCC?style=flat-square&logo=MySQL&logoColor=white"/>
  <img src="https://img.shields.io/badge/Firebase-blue?style=flat-square&logo=Firebase&logoColor=FFCA28"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Anaconda-e9e9e9?style=flat-square&logo=Anaconda&logoColor=44A833"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-FF9900?style=flat-square&logo=PyTorch&logoColor=EE4C2C"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>   

### 🚀 Deep Learning Model
<p>
  <img src="https://img.shields.io/badge/YOLO-0000FF?"/>
  <img src="https://img.shields.io/badge/I3D-FF3399?"/>
</p>

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; COCO Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Color Helmet and Vest Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; HMDB51 Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Something-Something-V2 Dataset <br/>
