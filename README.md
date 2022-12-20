## 📹 Intelligent CCTV for Port Safety
- _The PDF file is a paper of Intelligent CCTV for Port Safety published by KIPS(Korean Information Processing Society)._ <br/> <br/> <br/> 

### 1. &nbsp; Background of Development <br/> 
- _Recently, CCTV-related technologies have been used in various ways in our daily lives, such as security and safety accident prevention. However, traditional CCTV cameras record more unnecessary information than is needed when a problem occurs. In addition, it is difficult to fully recognize and judge the field situation only with existing CCTV cameras. The Korea Safety and Health Agency announced that six deaths occurred every year at domestic ports during 2011-2021. It shows the limitations that existing CCTV cameras cannot solve safety accidents and human casualties in domestic ports. In order to solve these limitations, I devised "Intelligent CCTV for Port Safety" that can quickly and accurately respond to dangerous situations while checking the site situation in real time._ <br/><br/><br/>

### 2. &nbsp; Project Introduction <br/> 

- _The project used a variety of object detection and action recognition models based on traditional computer vision technique and deep neural network._ <br/><br/>

- _Based on the various deep learning models mentioned above, the following algorithms are implemented :_ <br/>

  - _Object Tracking_
  - _Region of Interest (ROI)_
  - _Time Series Data Analysis_
  - _Measuring Distance Between Objects_ <br/><br/>
  
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

  - _Regions of interest (ROI) means the meaningful and important regions in the images._ <br/>
    - _ROI function eliminates unnecessary areas from the image, improving the processing speed and accuracy of object detection and action recognition._ <br/>
    - _The ROI function is implemented based on the binary mask technique in the image processing field._ <br/>
    - _In the mask image, pixels that belong to the ROI are set to 1 and pixels outside the ROI are set to 0._ <br/> 
    
  - _The user can set the ROI to a rectangular or polygonal shape, and object detection and action recognition are processed only within the Red ROI border._ <br/>
    - _Green ROI Border &nbsp; : &nbsp; specifying_ <br/>
    - _Yellow ROI Border &nbsp; : &nbsp; modifying_ <br/>
    - _Red ROI Border &nbsp; : &nbsp; setup complete_ <br/><br/><br/>
    
 - _**Intrusion and Loitering**_ <br/><br/>
<img src="https://github.com/qortmdgh4141/Intelligent_CCTV_for_Port_Safety/blob/main/image/intrusion_and_loitering.png?raw=true"  width="1280" height="240"> <br/>  

    - _Object Detection model detects human intrusions._ <br/>
    
    - _If the intruder stays for a long time, it is judged that intruder is  loitering._ <br/>
    
    - _Even if the intruder and loiterer appear again after being covered by another object or going out of the video, they are recognized as the same person because the DeepSort algorithm has been applied._ <br/>
      - _DeepSORT can be defined as the tracking algorithm which tracks objects not only based on the velocity and motion of the object but also the appearance of the object._ <br/>
      - _Intruder and loiterer are given a unique ID by applying the DeepSort algorithm._ <br/>
      - _Information of previous intruder and loiterer can be inquired with a unique ID._ <br/><br/><br/>



 -- 

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
