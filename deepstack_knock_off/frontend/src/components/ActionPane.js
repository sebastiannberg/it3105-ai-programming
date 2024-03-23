import { useState, useEffect, useCallback } from "react";
import Action from "./Action";
import axios from 'axios';


const ActionPane = ({ gameState, fetchGameState, onWinnerDetermined, onRoundWinners, onAiDecision, winner, roundWinners, aiDecision }) => {
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

  const fetchAIDecision = useCallback(async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/ai-decision")
      console.log(response.data)
      onAiDecision(response.data)
    } catch (error) {
      console.error("Failed to fetch AI decision:", error)
    }
  }, [onAiDecision])

  useEffect(() => {
    fetchLegalActions();
    const activePlayerName = Object.keys(gameState.active_player || {})[0];
    if (activePlayerName && activePlayerName.includes("AI")) {
      fetchAIDecision();
    }
  }, [gameState, fetchAIDecision]);

  const applyAIAction = useCallback((action) => {
    console.log("Applying AI action:", action);
    const payload = {
      name: action.name,
      player: action.player
    };

    axios.post("http://127.0.0.1:5000/apply-action", payload)
      .then(response => {
        console.log("AI action applied successfully:", response.data);
        if (response.data.winner) {
          // A game winner has been determined
          onWinnerDetermined(response.data.winner)
          setActions([])
          setSelectedAction(null)
        } else if (response.data.round_winners) {
          onRoundWinners(response.data.round_winners)
          setActions([]);
          setSelectedAction(null)
        } else {
          fetchGameState();
        }
        onAiDecision(null); // Properly reset aiDecision here if not done elsewhere
      })
      .catch(error => {
        console.error("Failed to apply AI action:", error);
      });
      // Reset selection after applying
      setSelectedAction(null);
    }, [fetchGameState, onAiDecision, onWinnerDetermined, onRoundWinners]);

  // Adjust the useEffect hook for applying AI decisions
  useEffect(() => {
    // Check if aiDecision is not null and has the necessary properties
    if (aiDecision && aiDecision.name) {
      const timer = setTimeout(() => {
        applyAIAction(aiDecision);
        onAiDecision(null); // Reset aiDecision to null after applying
      }, 3000); // Wait 3 seconds

      return () => clearTimeout(timer); // Clean up the timeout if the component unmounts
    }
  }, [aiDecision, onAiDecision, applyAIAction]); // Depend on aiDecision


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
        if (response.data.winner) {
          // A game winner has been determined
          onWinnerDetermined(response.data.winner)
          setActions([]);
          setSelectedAction(null);
        } else if (response.data.round_winners) {
          onRoundWinners(response.data.round_winners)
          setActions([]);
          setSelectedAction(null);
        } else {
          fetchGameState()
        }
      })
      .catch(error => {
        console.error("Failed to apply action:", error);
      });
    // Reset selection after applying
    setSelectedAction(null);
  };

  const buttonClass = selectedAction ? "poker-button" : "poker-button disabled";
  const shouldDisplayActions = !winner && !roundWinners.length;

  return (
    <div className="action-pane">
      <div className="scroll-box">
        <ul>
          {shouldDisplayActions && actions.map((action, index) => (
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
