import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SetupPage = () => {
  const navigate = useNavigate();
  const [config, setConfig] = useState("{}");
  const [rules, setRules] = useState("{}");
  const [error, setError] = useState("");

  const fetchPlaceholders = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/placeholders");
      setConfig(JSON.stringify(response.data.config, null, 4));
      setRules(JSON.stringify(response.data.rules, null, 4));
    } catch (error) {
      console.error('Failed to fetch placeholders:', error);
    }
  };

  useEffect(() => {
    fetchPlaceholders();
  }, []);

  const handleConfigChange = (event) => {
    setConfig(event.target.value);
  };

  const handleRulesChange = (event) => {
    setRules(event.target.value);
  };

  const formatAndSet = (jsonString, setterFunction) => {
    try {
      const parsedJson = JSON.parse(jsonString);
      setterFunction(JSON.stringify(parsedJson, null, 4));
    } catch (error) {
      setError("Failed to parse JSON. Please check the format.");
    }
  };

  const handleBlur = (setterFunction) => (event) => {
    formatAndSet(event.target.value, setterFunction);
  };

  const startGame = async () => {
    try {
      const parsedConfig = JSON.parse(config);
      const parsedRules = JSON.parse(rules);
      await axios.post("http://127.0.0.1:5000/start-game", { config: parsedConfig, rules: parsedRules });
      navigate("/board");
    } catch (error) {
      console.error('Failed to parse JSON or start game:', error);
      setError("Failed to parse JSON. Please check the format.");
    }
  };

  return (
    <div>
      <div className='setup-box'>
        <h1 className='title'>Poker Config</h1>
        <textarea
            value={config}
            onChange={handleConfigChange}
            onBlur={handleBlur(setConfig)}
            rows={10}
            cols={60}
          />
        <h1 className='title'>Poker Rules</h1>
        <textarea
          value={rules}
          onChange={handleRulesChange}
          onBlur={handleBlur(setRules)}
          rows={10}
          cols={60}
        />
        {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
        <button className='poker-button' onClick={startGame}>START GAME</button>
      </div>
    </div>
  );
};

export default SetupPage;
