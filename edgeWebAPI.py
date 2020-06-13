from appEdge import app
import config


"""
Initialize Edge API
"""

app.debug = config.DEBUG
app.run(host=config.HOST_EDGE, port=config.PORT_EDGE)