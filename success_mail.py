
import smtplib 
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
# start TLS for security 
s.starttls()   
# Authentication 
s.login("sender_mail", "yourpwd") 
# message to be sent 
message = "Hey Developer, your model is trained and giving good accuracy!"
# sending the mail 
s.sendmail("sender_mail", "developer_mail", message) 
# terminating the session 
print("Mail has been sent to developer.")
s.quit()
