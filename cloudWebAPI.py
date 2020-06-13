from appCloud import app
import config

app.debug = config.DEBUG
app.run(host=config.HOST_CLOUD, port=config.PORT_CLOUD)