## 📹 Intelligent CCTV for Port Safety
- _The PDF file is a paper of Intelligent CCTV for Port Safety published by KIPS(Korean Information Processing Society)._ <br/> <br/> <br/> 

### 1. &nbsp; Background of Development <br/> 
- _Recently, CCTV-related technologies have been used in various ways in our daily lives, such as security and safety accident prevention. However, traditional CCTV cameras record more unnecessary information than is needed when a problem occurs. In addition, it is difficult to fully recognize and judge the field situation only with existing CCTV cameras. The Korea Safety and Health Agency announced that six deaths occurred every year at domestic ports during 2011-2021. It shows the limitations that existing CCTV cameras cannot solve safety accidents and human casualties in domestic ports. In order to solve these limitations, I devised "Intelligent CCTV for Port Safety" that can quickly and accurately respond to dangerous situations while checking the site situation in real time._ <br/><br/><br/>

### 2. &nbsp; Project Introduction <br/> 

- _The project used a variety of object detection and action recognition models based on traditional computer vision technique and deep neural network._ <br/><br/>

- _Based on the various deep learning models mentioned above, the following algorithms are implemented :_ <br/>
  
  - _Region of Interest (ROI)_ <br/>
    - _Regions of interest (ROI) means the meaningful and important regions in the images._ <br/>
    - _The ROI algorithm is implemented based on the binary mask technique in the image processing field._ <br/>
    - _In the mask image, pixels that belong to the ROI are set to 1 and pixels outside the ROI are set to 0._ <br/> 
    
  - _DeepSort_
    - _DeepSort is the tracking algorithm which tracks objects not only based on the velocity and motion of the object but also the appearance of the object._ <br/>
    
  - _Time Series Data Analysis_ <br/>
  
  - _Measuring Distance Between Objects_ <br/>
    - _The algorithm detects and analyzes low-risk and high-risk states by measuring the distance between the bounding box centers of objects._ <br/>
    - _When measuring distance, measure the distance between all objects detected in the image by reflecting the features of the 'complete graph'._<br/>
    
    
    
  
- _Through the algorithms mentioned above, the following events are detected and analyzed :_ <br/>

  - _Intrusion and Loitering_
  - _Risk of Access to Restricted Areas_
  - _Risk of Collision Between Workers_
  - _Risk of Not Wearing Worker Safety Equipment_
  - _Fire Accident_
  - _Act of Violence_
  - _Act of Falling_
  - _Act of Smoking_ <br/><br/>

- _The analyzed information is stored in the database, and the administrator can quickly grasp the field situation through text and graph-type information provided in real time._ <br/>

  - _Field situation information can be checked not only on PC but also on smartphone._ <br/><br/>

- _If these functions are applied to the integrated control center in the port, it will be possible to build an intelligent infrastructure that can efficiently manage and operate. In other words, it can contribute to the introduction of a smart port monitoring system in the future._ <br/><br/><br/>
  
### 3. &nbsp; Main Function <br/><br/>
- _**Region of Interest (ROI)**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/region_of_interest_(roi).png?raw=true"  width="1280" height="340"> <br/>

  - _The administrator can set the ROI to a rectangle or polygon shape by dragging or clicking the mouse._ <br/>  
  
  - _Object detection is processed only within the Red ROI Border._ 
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
      - _When the intruder reappears, it is recognized as the same person by the DeepSort algorithm and given a unique ID number previously given._ <br/>
      - _It then applies the previous information about the intruder by querying the unique ID number from the database._ <br/>
      
   - _Through this, the administrator can individually detect and analyze whether many people in the port are intruding and loitering._ <br/><br/><br/>

 - _**Risk of Access to Restricted Areas**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_access_to_restricted_areas.png?raw=true"  width="1280" height="230"> <br/>  

    - _The administrator can set the restricted area to a rectangle shape by dragging the mouse._ <br/>
        - _When the restricted area setting is completed, the restricted area is displayed as a white bounding box._ <br/> 
        
    - _Based on the algorithm of Measuring Distance Between Objects, when a person approaches a restricted area, a warning or dangerous message is sent to the administrator._ <br/>
        - Low-Risk States &nbsp;&nbsp; : &nbsp; The border of the bounding box is displayed in yellow.  <br/>        
        - High-Risk States &nbsp; : &nbsp; The border of the bounding box is displayed in red. <br/>
        
    - _Through this, the administrator can proactively block people from entering restricted areas within the port._ <br/><br/><br/>
        
 - _**Risk of Collision Between Workers**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_collision_between_workers.png?raw=true"  width="1280" height="230"> <br/>  

    - _Based on the algorithm of Measuring Distance Between Objects, if there is a possibility that the safe distance between workers is not secured, a warning or dangerous message is sent to the administrator._ <br/>
        - _Low-Risk States &nbsp;&nbsp; : &nbsp; The safe distance between workers is displayed in yellow._
        - _High-Risk States &nbsp; : &nbsp; The safe distance between workers is displayed in red._  <br/>
     
    - _Through this, the administrator can prevent collision accidents caused by the failure of workers to secure a safe distance in a dense space._ <br/><br/><br/>
        
 - _**Risk of Not Wearing Worker Safety Equipment**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_not_wearing_worker_safety_equipment.png?raw=true"  width="1280" height="440"> <br/>  

    - _First, the object detection model detects a worker, a safety helmet, and a safety vest._ <br/>
    
    - _Next, based on the algorithm of Measuring Distance Between Objects, it analyzes whether the worker wears safety equipment._ <br/>
    
    - _If the worker is not wearing safety equipment, a warning or dangerous message is sent to the administrator._ <br/>
         - Low-Risk States &nbsp;&nbsp; : &nbsp; The Δ symbol is displayed in yellow in the bounding box.<br/>
         - High-Risk States &nbsp; : &nbsp; The X symbol is displayed in red in the bounding box. <br/>
         
    - _Through this, the administrator can prevent safety accidents caused by workers not wearing safety equipment at the work site._ <br/><br/><br/>
 
 - _**Act of Violence**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/risk_of_collision_between_workers.png?raw=true"  width="1280" height="230"> <br/>  

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
  <img src="https://img.shields.io/badge/YOLO-black?&logo=YOLO&logoColor=00FFFF"/>
  <img src="https://img.shields.io/badge/I3D-FF3399?"/>
</p>

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; COCO Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Color Helmet and Vest Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; HMDB51 Dataset <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Something-Something-V2 Dataset <br/>
