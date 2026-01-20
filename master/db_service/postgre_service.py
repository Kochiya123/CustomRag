from master.db_module.mysql_connect import mysql_connect


class MysqlService(object):
    def __init__(self, db_service=None):
        self.conn = mysql_connect()
        self.cursor = self.conn.cursor()

    def get_tables_schema(self):
        table_query = "SELECT TABLE_SCHEMA AS SchemaName, TABLE_NAME AS TableName, COLUMN_NAME AS ColumnName, DATA_TYPE AS DataType FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA != 'sys' ORDER BY TABLE_SCHEMA, TABLE_NAME"
        self.cursor.execute(table_query)
        table_result = self.cursor.fetchall()
        return table_result

    def get_relational_schema(self):
        relational_query = "SELECT TABLE_NAME AS foreign_table, COLUMN_NAME AS foreign_column, CONSTRAINT_NAME AS constraint_name, REFERENCED_TABLE_NAME AS referenced_table, REFERENCED_COLUMN_NAME AS referenced_column FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE WHERE REFERENCED_TABLE_SCHEMA = 'flowerplus' AND REFERENCED_TABLE_NAME IS NOT NULL"
        self.cursor.execute(relational_query)
        relation_result = self.cursor.fetchall()
        return relation_result

    def get_product_on_query(self, query):
        self.cursor.execute(query)
        product_result_list = self.cursor.fetchall()
        print(product_result_list)
        return product_result_list

    def get_product_on_id(self, result_list):
        if not result_list:
            return []

        # Extract IDs from result_list
        product_ids = [result[0] for result in result_list]

        # Use IN clause for single query (more efficient)
        placeholders = ','.join(['%s'] * len(product_ids))
        query = f"""
            SELECT id, name, price, stock, description, is_active 
            FROM flowerplus.products 
            WHERE id IN ({placeholders}) 
            AND product_type = 0 
            AND is_custom = 0
        """

        self.cursor.execute(query, product_ids)
        rows = self.cursor.fetchall()

        # Convert all rows to dictionaries with column names
        columns = [col[0] for col in self.cursor.description]
        product_result_list = [dict(zip(columns, row)) for row in rows]

        return product_result_list

    def return_connection(self):
        return self.conn

    def close_connection(self):
        self.cursor.close()
        self.conn.close()
        return 0
