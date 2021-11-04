import paramiko

# 建立一个sshclient对象
ssh = paramiko.SSHClient()
# 允许将信任的主机自动加入到host_allow 列表，此方法必须放在connect方法的前面
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 调用connect方法连接服务器
ssh.connect(hostname=xxx, port=xxx, username=xxx, password=xxx)
# 执行命令
# comds是相关的指令列表，需要注意的是一连串的指令需要用;分割开，不然不会连续执行，
# 比如cd /data/;python main.py
for command in comds:
    stdin, stdout, stderr = ssh.exec_command(command)
    # stdin, stdout, stderr = ssh.exec_command('ls')
    # 结果放到stdout中，如果有错误将放到stderr中
    print(stdout.read().decode())
# 关闭连接
ssh.close()
