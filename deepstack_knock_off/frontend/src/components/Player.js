

const Player = ({ name, chips, hand, bet }) => {

  return (
    <div className="player">
      <h3>{name}</h3>
      <p>Chips: {chips}</p>
      <p>Current Bet: {bet}</p>
      <p>{hand}</p>
    </div>
  )
};

export default Player;