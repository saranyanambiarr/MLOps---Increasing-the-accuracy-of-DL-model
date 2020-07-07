
import smtplib 
# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 
# start TLS for security 
s.starttls()   
# Authentication 
s.login("sharanya8095@gmail.com", "lEgAcY123_21") 
# message to be sent 
message = "Hey Developer, your model is trained and giving good accuracy!"
# sending the mail 
s.sendmail("sharanya8095@gmail.com", "nambiar.saranya98@gmail.com", message) 
# terminating the session 
print("Mail has been sent to developer.")
s.quit()
