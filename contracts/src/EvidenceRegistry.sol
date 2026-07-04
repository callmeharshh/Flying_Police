// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EvidenceRegistry
/// @notice Tamper-proof anchoring of Flying Police drone security evidence on Monad.
///         Full evidence JSON is stored on IPFS (Lighthouse). The keccak256 hash,
///         frameId, location, message, and IPFS CID are anchored on-chain.
contract EvidenceRegistry {
    struct Record {
        uint64 anchoredAt;
        uint64 frameId;
        uint8 severity;
        address anchoredBy;
        string location;
        string message;
        string ipfsCid;
    }

    address public owner;

    mapping(bytes32 => Record) private _records;

    uint256 public totalAnchored;

    event EvidenceAnchored(
        bytes32 indexed evidenceHash,
        uint64 indexed frameId,
        uint8 severity,
        uint64 anchoredAt,
        address indexed anchoredBy,
        string location,
        string message,
        string ipfsCid
    );

    event OwnershipTransferred(address indexed from, address indexed to);

    error NotOwner();
    error AlreadyAnchored(bytes32 evidenceHash);
    error LengthMismatch();
    error ZeroAddress();

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    /// @notice Anchor evidence with on-chain metadata and an IPFS content id.
    function anchor(
        bytes32 evidenceHash,
        uint64 frameId,
        uint8 severity,
        string calldata location,
        string calldata message,
        string calldata ipfsCid
    ) external onlyOwner {
        _anchor(evidenceHash, frameId, severity, location, message, ipfsCid);
    }

    /// @notice Anchor many pieces of evidence in one transaction.
    function batchAnchor(
        bytes32[] calldata evidenceHashes,
        uint64[] calldata frameIds,
        uint8[] calldata severities,
        string[] calldata locations,
        string[] calldata messages,
        string[] calldata ipfsCids
    ) external onlyOwner {
        if (
            evidenceHashes.length != frameIds.length
                || frameIds.length != severities.length
                || severities.length != locations.length
                || locations.length != messages.length
                || messages.length != ipfsCids.length
        ) {
            revert LengthMismatch();
        }
        for (uint256 i = 0; i < evidenceHashes.length; i++) {
            _anchor(
                evidenceHashes[i],
                frameIds[i],
                severities[i],
                locations[i],
                messages[i],
                ipfsCids[i]
            );
        }
    }

    function _anchor(
        bytes32 evidenceHash,
        uint64 frameId,
        uint8 severity,
        string calldata location,
        string calldata message,
        string calldata ipfsCid
    ) private {
        if (_records[evidenceHash].anchoredAt != 0) revert AlreadyAnchored(evidenceHash);
        uint64 ts = uint64(block.timestamp);
        _records[evidenceHash] = Record({
            anchoredAt: ts,
            frameId: frameId,
            severity: severity,
            anchoredBy: msg.sender,
            location: location,
            message: message,
            ipfsCid: ipfsCid
        });
        unchecked {
            totalAnchored++;
        }
        emit EvidenceAnchored(
            evidenceHash, frameId, severity, ts, msg.sender, location, message, ipfsCid
        );
    }

    function verify(bytes32 evidenceHash) external view returns (bool exists, uint64 anchoredAt) {
        Record memory r = _records[evidenceHash];
        return (r.anchoredAt != 0, r.anchoredAt);
    }

    function getRecord(bytes32 evidenceHash) external view returns (Record memory) {
        return _records[evidenceHash];
    }

    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}
