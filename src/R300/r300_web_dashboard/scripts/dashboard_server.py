#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R300 Web 上位机静态文件服务器。

这个节点只负责把 r300_web_dashboard/www 目录通过 HTTP 发给浏览器。
真正的 ROS 话题通讯由浏览器端通过 rosbridge websocket 完成；
图像流由 web_video_server 提供。
"""

from __future__ import print_function

import os
import sys
import threading
try:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
except ImportError:  # Python2 fallback, normally not used on Noetic
    from SimpleHTTPServer import SimpleHTTPRequestHandler
    from SocketServer import ThreadingMixIn
    from BaseHTTPServer import HTTPServer
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

import rospy


def _package_www_dir():
    """Return package www directory; support both installed and source-tree use."""
    try:
        import rospkg
        pkg_dir = rospkg.RosPack().get_path("r300_web_dashboard")
        www_dir = os.path.join(pkg_dir, "www")
        if os.path.isdir(www_dir):
            return www_dir
    except Exception:
        pass
    # fallback: scripts/../www
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "www"))


class NoCacheHandler(SimpleHTTPRequestHandler):
    """Disable browser cache so editing app.js/config.json takes effect after refresh."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        SimpleHTTPRequestHandler.end_headers(self)

    def log_message(self, fmt, *args):
        rospy.loginfo("dashboard_server: " + fmt, *args)


def main():
    rospy.init_node("r300_web_dashboard_server", anonymous=False)
    port = int(rospy.get_param("~port", 8090))
    bind = rospy.get_param("~bind", "0.0.0.0")
    www_dir = rospy.get_param("~www_dir", _package_www_dir())

    if not os.path.isdir(www_dir):
        rospy.logerr("Web directory not found: %s", www_dir)
        sys.exit(1)

    os.chdir(www_dir)
    httpd = ThreadingHTTPServer((bind, port), NoCacheHandler)

    def serve():
        rospy.loginfo("R300 Web dashboard serving %s at http://%s:%d", www_dir, bind, port)
        httpd.serve_forever()

    t = threading.Thread(target=serve)
    t.daemon = True
    t.start()

    rospy.loginfo("Open browser: http://<robot-ip>:%d", port)
    rospy.spin()
    httpd.shutdown()


if __name__ == "__main__":
    main()
