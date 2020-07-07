
import smtplib 
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
# start TLS for security 
s.starttls()   
# Authentication 
s.login("sharanya8095@gmail.com", "lEgAcY123_21") 
# message to be sent 
message = "Hey Developer, you need to check your code once. The accuracy of your model is not so good."
# sending the mail 
s.sendmail("sharanya8095@gmail.com", "nambiar.saranya98@gmail.com", message) 
# terminating the session 
print("Mail has been sent to developer.")
s.quit()