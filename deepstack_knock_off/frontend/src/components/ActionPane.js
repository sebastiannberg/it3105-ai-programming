import { useState, useEffect } from "react";
import Action from "./Action";
import axios from 'axios';


const ActionPane = ({ gameState, fetchGameState, onWinnerDetermined, onRoundWinner }) => {
  const [actions, setActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);

  const fetchLegalActions = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/legal-actions");
      setActions(response.data);
    } catch (error) {
      console.error('Failed to fetch legal actions:', error);
    }
  };

  useEffect(() => {
    fetchLegalActions();
  }, [gameState]);

  const handleActionClick = (action) => {
    setSelectedAction(action);
  };

  const applySelectedAction = () => {
    console.log("Applying action:", selectedAction);

    const payload = {
      name: selectedAction.name,
      player: selectedAction.player
    };

    axios.post("http://127.0.0.1:5000/apply-action", payload)
      .then(response => {
        console.log("Action applied successfully:", response.data);
        fetchGameState()

        if (response.data.winner) {
          onWinnerDetermined(response.data.winner)
          setActions([]);
          setSelectedAction(null);
        } else if (response.data.round_winner) {
          onRoundWinner(response.data.round_winner)
          setActions([]);
          setSelectedAction(null);
        }
      })
      .catch(error => {
        console.error("Failed to apply action:", error);
      });

    // Reset selection after applying
    setSelectedAction(null);
  };

  const buttonClass = selectedAction ? "poker-button" : "poker-button disabled";

  return (
    <div className="action-pane">
      <div className="scroll-box">
        <ul>
          {actions.map((action, index) => (
            <Action
              key={index}
              action={action}
              onClick={() => handleActionClick(action)}
              isSelected={selectedAction === action}
            />
          ))}
        </ul>
      </div>
      <button className={buttonClass} onClick={applySelectedAction} disabled={!selectedAction || actions.length === 0}>Apply</button>
    </div>
  );
};

export default ActionPane;