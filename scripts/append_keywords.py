"""Script to append new keyword categories to keywords.csv"""
import os

new_rows = (
    '106,junk_name_patterns,email_extractor,'
    '"recruiting,booking,noreply,postmaster,talent,scheduling,eximius,hr,hiring,'
    'operations,notifications,alerts,staffing,sourcing,support,admin,careers,'
    'apply,jobs,info,team,sales,marketing,service,services,helpdesk,system,'
    'office,reception,contact,inquiries,engagement,pipeline,screening,placements,'
    'resources,acquisition,onboarding,outreach,community,partnerships",'
    'contains,block,200,'
    'Single role/noun words that are never valid recruiter person names,'
    '1,2026-02-27T08:00:00,2026-02-27T08:00:00\n'

    '107,recruiter_email_signals,email_extractor,'
    '"i have an opening,we are currently hiring,on behalf of our client,'
    'your profile on linkedin,your background,contract position,full-time role,'
    'permanent role,direct client,end client,looking for a candidate,'
    'suitable candidate,job opportunity,open position,immediate opening,'
    'direct placement,staffing agency,i am a recruiter,my client is,'
    'we are hiring,we have an urgent,urgent requirement,c2c opportunity,'
    'w2 opportunity,1099 opportunity,reach out regarding,came across your profile,'
    'your experience matches,great fit for this role,excited to share,'
    'hope you are open to,happy to connect,let me know your interest,'
    'please share your updated resume,kindly share your resume",'
    'contains,allow,200,'
    "Body phrases that strongly confirm recruiter intent,"
    '1,2026-02-27T08:00:00,2026-02-27T08:00:00\n'

    '108,non_recruiter_body_signals,email_extractor,'
    '"unsubscribe,click here to unsubscribe,free trial,webinar,newsletter,'
    'download now,limited time offer,sign up now,register now,new features,'
    'product launch,case study,ebook,whitepaper,learn more,book a demo,'
    'get started for free,special offer,exclusive deal,you are receiving this,'
    'if you no longer wish,to opt out,manage preferences,view in browser,'
    'this message was sent to,powered by mailchimp,sent via sendgrid,'
    'update your preferences,you subscribed",'
    'contains,block,200,'
    'Marketing/spam body phrases that disqualify email before NLP runs,'
    '1,2026-02-27T08:00:00,2026-02-27T08:00:00\n'
)

csv_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'keywords.csv')
csv_path = os.path.normpath(csv_path)

with open(csv_path, 'a', encoding='utf-8', newline='') as f:
    f.write(new_rows)

print(f"Appended 3 new keyword categories to {csv_path}")
