# sql相关的一些
## MySql
### 利用正则表达式查询
```sql
select * from xxx where content REGEXP "(淘宝|天猫|(?<![0-9])1688(?![0-9]))"
```
### case when的使用
```sql
select case xxx when xxx = ? then ? when xxx = ? then ? else ? end as ? from xxx
```
