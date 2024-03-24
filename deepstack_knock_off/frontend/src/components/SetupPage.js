import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SetupPage = () => {
  const navigate = useNavigate();
  const [config, setConfig] = useState({});
  const [rules, setRules] = useState({});

  const fetchPlaceholders = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/placeholders");
      console.log(response.data)
      setConfig(response.data.config);
      setRules(response.data.rules);
    } catch (error) {
      console.error('Failed to fetch placeholders:', error);
    }
  };

  useEffect(() => {
    fetchPlaceholders();
  }, []);

  const handleConfigChange = (event) => {
    setConfig(JSON.parse(event.target.value));
  };

  const handleRulesChange = (event) => {
    setRules(JSON.parse(event.target.value));
  };

  const startGame = async () => {
    try {
      await axios.post("http://127.0.0.1:5000/start-game", { config, rules });
      navigate("/board")
    } catch (error) {
      console.error('Failed to start game or parse:', error);
    }
  };

  return (
    <div>
      <div className='setup-box'>
        <h1 className='title'>Poker Config</h1>
        <textarea
            value={JSON.stringify(config, null, 4)}
            onChange={handleConfigChange}
            rows={10}
            cols={60}
          />
        <h1 className='title'>Poker Rules</h1>
        <textarea
          value={JSON.stringify(rules, null, 4)}
          onChange={handleRulesChange}
          rows={10}
          cols={60}
        />
      <button className='poker-button' onClick={startGame}>START GAME</button>
      </div>
    </div>
  );
};

export default SetupPage;
