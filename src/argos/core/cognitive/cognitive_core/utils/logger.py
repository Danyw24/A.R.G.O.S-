class Log:
  def status(self, msg):
    print("\033[0;96m[~]\033[0m %s" % msg)
  def success(self, msg):
    print("\033[0;92m[+]\033[0m %s" % msg)
  def error(self, msg):
    print("\033[0;91m[!]\033[0m %s" % msg)
  def debug(self, msg):
    print("\033[0;37m[.]\033[0m %s" % msg)
  def notice(self, msg):
    print("\033[0;93m[?]\033[0m %s" % msg)
  def info(self, msg):
    print("\033[0;94m[*]\033[0m %s" % msg)
  def enum(self, index, msg):
    print("\033[0;94m<\033[0m%s\033[0;94m>\033[0m %s" % (index, msg))
  def warning(self, msg):
    print("\033[0;93m[!]\033[0m %s" % msg)

