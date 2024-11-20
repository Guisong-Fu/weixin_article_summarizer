import pytest
from unittest.mock import Mock, patch
from src.weixin_summarizer.llm_handler import LLMHandler

@pytest.mark.asyncio
async def test_llm_initialization():
    """Test LLM initialization with OpenAI provider."""
    with patch.dict('os.environ', {'LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'}):
        handler = LLMHandler()
        assert handler.provider == 'openai'
        assert handler.model_name == 'gpt-3.5-turbo'

@pytest.mark.asyncio
async def test_generate_completion():
    """Test completion generation."""
    with patch.dict('os.environ', {'LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'}):
        handler = LLMHandler()
        
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "Test completion"
        handler.llm = Mock()
        handler.llm.ainvoke = Mock(return_value=mock_response)
        
        result = await handler.generate_completion(
            "Test prompt {var}",
            {"var": "test"}
        )
        
        assert result == "Test completion"
        handler.llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_unsupported_provider():
    """Test initialization with unsupported provider."""
    with patch.dict('os.environ', {'LLM_PROVIDER': 'unsupported'}):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMHandler() 