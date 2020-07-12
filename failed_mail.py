
import smtplib 
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
# start TLS for security 
s.starttls()   
# Authentication 
s.login("sender_mail", "your_pwd") 
# message to be sent 
message = "Hey Developer, you need to check your code once. The accuracy of your model is not so good."
# sending the mail 
s.sendmail("sender_mail", "developer_mail", message) 
# terminating the session 
print("Mail has been sent to developer.")
s.quit()
