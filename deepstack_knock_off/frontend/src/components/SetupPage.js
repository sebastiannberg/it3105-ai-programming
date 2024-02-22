import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SetupPage = () => {
  const navigate = useNavigate();
  const [rules, setRules] = useState("");

  const fetchRules = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/rules");
      setRules(JSON.stringify(response.data, null, 2)); // Format with newlines and indentation
    } catch (error) {
      console.error('Failed to fetch rules:', error);
    }
  };

  useEffect(() => {
    fetchRules();
  }, []);

  const handleRulesChange = (event) => {
    setRules(event.target.value);
  };

  const startGame = async () => {
    try {
      const updatedRules = JSON.parse(rules);
      await axios.post("http://127.0.0.1:5000/start-game", updatedRules);
      navigate("/board")
    } catch (error) {
      console.error('Failed to start game or parse rules:', error);
    }
  };

  return (
    <div>
      <h1 className='title'>Poker Rules</h1>
      <div className='setup-box'>
        <textarea
          value={rules}
          onChange={handleRulesChange}
          rows={25}
          cols={60}
        />
        <button className='poker-button' onClick={startGame}>START GAME</button>
      </div>
    </div>
  );
};

export default SetupPage;
