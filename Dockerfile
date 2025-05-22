FROM mysql:8.3.0

# 환경 변수 설정
ENV MYSQL_ROOT_PASSWORD=rootpassword
ENV MYSQL_DATABASE=knuckle
ENV MYSQL_USER=appuser
ENV MYSQL_PASSWORD=apppassword

# MySQL 설정 파일 복사 (옵션)
# COPY my.cnf /etc/mysql/conf.d/

# 초기화 스크립트 복사
COPY init-schema.sql /docker-entrypoint-initdb.d/

# 포트 노출
EXPOSE 3306

# MySQL 서버 시작
CMD ["mysqld"]