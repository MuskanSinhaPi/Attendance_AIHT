# Attendance_AIHT
Real time face recognising attendance management system <br>
<ul> <li> Modify the WiFi SSID and Password in the esp32 cam webserver ino code </li>
<li> add files to the images folder </li>
<li> change the path name in the python script for the esp32cam webserver and the images folder </li>
<li> copy path name of the desired directory in which the attendance.csv file is to be created </li>
<li> the attendance.csv file is created in that directory with the system date and attendance is updated with the name, respective Roll Number, and timestamp of detection. </li>

## Bug Resolutions, Functionalities to add in the Future
• Current code doesn't update attendance for someone whose attendance was marked for the previous day (misinterprets it as redundant logging)
need to reset log for each new day
<br><br>
• Functionality for liveness detection (to prevent proxies)