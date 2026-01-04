
import logging
import sys
import contextvars
from logger import logs, set_context, scoped_context

# Mock context object
class MockContext:
    def __init__(self, log_id):
        self.log_id = log_id

def test_logging():
    print("Testing logging with context...")
    
    # 1. No context
    logs.info("Message without context (should have log_id='-')")
    
    # 2. Key-value context (dict)
    ctx_dict = {"log_id": "DICT-123"}
    set_context(ctx_dict)
    logs.info("Message with dict context")
    
    # Reset context? No explicit reset in my code, but let's try another one.
    
    # 3. Object context
    ctx_obj = MockContext("OBJ-456")
    set_context(ctx_obj)
    logs.info("Message with object context")
    
    # 4. Scoped context
    with scoped_context({"log_id": "SCOPED-789"}):
        logs.info("Message inside scoped context")
        
    # 5. Back to previous? 
    # The scoped_context implementation resets to the token. 
    # But wait, set_context returns a Token.
    # scoped_context calls set_context (which returns a token), then yields, then resets(token).
    # So it should restore whatever was there before?
    # No, set_context sets the value.
    # So resetting the token restores the PREVIOUS value.
    # Let's verify.
    logs.info("Message after scoped context (should be OBJ-456)")

if __name__ == "__main__":
    test_logging()
