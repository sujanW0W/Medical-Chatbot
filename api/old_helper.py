from api.database import execute_statement
from api.types import *
from src.orchestrator import graph


def create_session(conn):
    try:
        create_session = """
INSERT INTO SESSIONS
VALUES(NULL, :session_name);
"""
        params = {
            "session_name": "New session"
        }
        execute_statement(conn, create_session, params)
        statement = """
SELECT last_insert_rowid();
"""
        res = execute_statement(conn, statement)

        return res.fetchone()[0]

    except Exception as e:
        print(e)
        raise Exception(e)


def generate_title(conn, session_id, user_msg, ai_msg):
    try:
        prompt = f"""
Generate a short 3-6 word title for this conversation.

User: {user_msg}

Assistant: {ai_msg}

Return only the title.
"""

        response = graph.invoke({
            "messages": [prompt]
        })

        title = response["messages"][-1].content

        rename_session_name(conn, session_id, title)

    except Exception as e:
        print(e)
        raise Exception(e)


def add_conversation(conn, session_id: int, message: Message):
    try:
        statement = """
INSERT INTO CONVERSATIONS
VALUES(NULL, :role, :content, :session_id);
"""
        params = {
            **message,
            "session_id": session_id
        }
        execute_statement(conn, statement, params)

    except Exception as e:
        print(e)
        raise Exception(e)


def fetch_all_conversations(conn, session_id):
    try:
        statement = f"""
SELECT * FROM CONVERSATIONS WHERE session_id=:session_id;
"""
        params = {
            "session_id": session_id
        }
        res = execute_statement(conn, statement, params)
        return res.fetchall()

    except Exception as e:
        print(e)
        raise Exception(e)


def fetch_sessions(conn):
    try:
        statment = f"""
SELECT * FROM SESSIONS;
"""
        res = execute_statement(conn, statment)

        return res.fetchall()

    except Exception as e:
        print(e)
        raise Exception(e)


def rename_session_name(conn, session_id: int, new_name: str):
    try:
        statement = f"""
UPDATE SESSIONS
SET session_name=:new_name
WHERE id=:session_id;
"""

        params = {
            "new_name": new_name,
            "session_id": session_id
        }

        execute_statement(conn, statement, params)

    except Exception as e:
        print(e)
        raise Exception(e)


def delete_session_func(conn, session_id: int):
    try:
        statement = f"""
DELETE FROM SESSIONS
WHERE id=:session_id;
"""

        params = {
            "session_id": session_id
        }

        execute_statement(conn, statement, params)

    except Exception as e:
        print(e)
        raise Exception(e)
