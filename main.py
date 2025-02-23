from flask import Flask
from app import views

app = Flask(__name__)

app.add_url_rule('/video_feed', 'video_feed', views.video_feed)
app.add_url_rule('/stop_video_feed', 'stop_video_feed', views.stop_video_feed)
app.add_url_rule('/app/gender/upload_video/', 'video_upload', view_func=views.video_upload, methods=['GET', 'POST'])

app.add_url_rule(rule='/',endpoint='home',view_func=views.index)
app.add_url_rule(rule='/app/',endpoint='app',view_func=views.app)
app.add_url_rule(rule='/app/gender/',
                 endpoint='gender',
                 view_func=views.genderapp,
                 methods=['GET','POST'])

if __name__ == "__main__":
    app.run(debug=True)
